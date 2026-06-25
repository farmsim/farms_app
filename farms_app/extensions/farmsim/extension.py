""" Main FARMSIM extension """

import os
import sys
import time

from farms_app.console import console
from farms_app.core.config_editor import ConfigEditorWindow
from farms_app.core.extension import Extension
from farms_app.core.widget import PlaybackState, SimulationToolbar
from farms_app.extensions.farmsim.data_registry import build_registry
from farms_app.extensions.farmsim.windows.mujoco_viewport import \
    MuJoCoViewportWindow
from farms_app.extensions.farmsim.windows.network_visualizer import \
    NetworkVisualizerWindow
from farms_app.extensions.farmsim.windows.properties import PropertiesWindow
from farms_app.plots.data_registry import DataRegistry
from farms_app.plots.plot_window import (PlotConfig, PlotWindow,
                                         PlotWindowConfig, SignalStyle)
from farms_core import pylog
from farms_core.sensors.data import SensorsData
from farms_core.experiment.options import ExperimentOptions
from farms_core.simulation.options import Simulator
from farms_sim.simulation import simulation_setup
from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd


class FARMSIMExtension(Extension):

    _PLOT_WINDOW_COUNTER = 0

    def __init__(self):
        super().__init__(name="FARMSIM")
        self.hooks.add("pre_substep")
        self.hooks.add("post_substep")
        self.hooks.add("create_default_plots")
        self.sim = None
        self.registry = DataRegistry()

        # Playback
        self.playback_state = PlaybackState.STOPPED
        self.playback_speed = 1.0
        self._dt_remainder = 0.0
        self._last_update_time = time.perf_counter()
        self._view_offset = 0  # 0 = live, negative = looking at history

        # Toolbar
        self.toolbar = SimulationToolbar()
        self.toolbar.on_play = self._play
        self.toolbar.on_pause = self._pause
        self.toolbar.on_stop = self._stop
        self.toolbar.on_step_fwd = lambda: self._step(1)
        self.toolbar.on_step_back = lambda: self._step(-1)
        self.toolbar.on_step_fwd_many = lambda: self._step(10)
        self.toolbar.on_step_back_many = lambda: self._step(-10)
        self.toolbar.on_speed_change = self._set_speed
        self.toolbar.on_scrub = self._scrub
        self.toolbar.on_record = self._toggle_record

        # Recording
        self._recording = False
        self._record_data = None      # ExperimentData for recording
        self._record_index = 0
        self._record_duration = 1000  # iterations
        self._record_path = ""
        self._show_record_popup = False

        # Windows
        self._mujoco_win = MuJoCoViewportWindow(self)
        self.register_window(self._mujoco_win)
        self._properties_win = PropertiesWindow(self)
        self.register_window(self._properties_win)
        self._config_win = ConfigEditorWindow(self, name="Config")
        self.register_window(self._config_win)
        self._network_vis_win = NetworkVisualizerWindow(self)
        self.register_window(self._network_vis_win)
        self._bottom_dock_id = 0
        self._new_plot_name = ""
        self._show_new_plot_popup = False

    def on_enable(self):
        from farms_app.core import layout
        if self.dockspace_id and layout.is_first_use():
            ds = self.dockspace_id
            left, rest = layout.split(ds, imgui.Dir.left, 0.2)
            self._bottom_dock_id, center = layout.split(rest, imgui.Dir.down, 0.25)
            layout.dock_window(self._config_win.window_id, left)
            layout.dock_window(self._properties_win.window_id, left)
            layout.dock_window(self._mujoco_win.window_id, center)
            layout.dock_window(self._network_vis_win.window_id, center)
            layout.finish(ds)

    # Data accessors
    @property
    def task(self):
        """The simulation task, or None if no sim is loaded."""
        if self.sim is None:
            return None
        return self.sim.task

    @property
    def farms_data(self):
        """AnimatData for the first animat, or None."""
        if self.sim is None:
            return None
        return self.sim.task.data.animats[0]

    @property
    def view_iteration(self):
        """The iteration index for rendering (accounts for scrub offset)."""
        if self.sim is None:
            return 0
        return self.task.iteration + self._view_offset

    @property
    def network(self):
        """Network from the first task extension, or None."""
        if self.sim is None:
            return None
        try:
            return self.sim.task.extensions[0].network
        except (IndexError, AttributeError):
            return None

    # Playback controls
    def _play(self):
        if self.sim is None:
            return
        self.playback_state = PlaybackState.PLAYING
        self._last_update_time = time.perf_counter()
        self._dt_remainder = 0.0
        self._view_offset = 0

    def _pause(self):
        self.playback_state = PlaybackState.PAUSED

    def _stop(self):
        self.playback_state = PlaybackState.STOPPED
        self._dt_remainder = 0.0

    def _set_speed(self, speed):
        self.playback_speed = speed

    def _scrub(self, offset):
        """Set view offset into the ring buffer. Auto-pauses playback."""
        if self.sim is None:
            return
        if self.playback_state == PlaybackState.PLAYING:
            self.playback_state = PlaybackState.PAUSED
        task = self.task
        max_back = -min(task.iteration, task.buffer_size)
        self._view_offset = max(max_back, min(0, int(offset)))

    def _toggle_record(self):
        """Toggle recording. Opens config popup or stops active recording."""
        if self.sim is None:
            return
        if self._recording:
            self._stop_recording()
        else:
            # Default save path next to experiment file
            if not self._record_path:
                exp_dir = os.path.dirname(getattr(self, '_experiment_path', '') or '')
                self._record_path = os.path.join(exp_dir, "recording.hdf5") if exp_dir else "recording.hdf5"
            self._show_record_popup = True

    def _start_recording(self):
        """Create recording ExperimentData (with network log if available), start capturing."""
        import numpy as np
        from farms_core.model.data import AnimatData
        from farms_core.simulation.data import SimulationData
        from farms_core.experiment.data import ExperimentData
        sensors = self.farms_data.sensors
        network_log = None
        network = self.network
        if network is not None:
            from farms_network.core.data import NetworkLog
            opts = network.options
            orig_buf = opts.logs.buffer_size
            opts.logs.buffer_size = self._record_duration
            network_log = NetworkLog.from_options(opts)
            opts.logs.buffer_size = orig_buf
        timestep = self.task.timestep
        self._record_data = ExperimentData(
            times=np.arange(self._record_duration) * timestep,
            timestep=timestep,
            simulation=SimulationData.from_size(self._record_duration),
            animats=[AnimatData(
                sensors=SensorsData.from_names(
                    buffer_size=self._record_duration,
                    links_names=sensors.links.names,
                    joints_names=sensors.joints.names,
                    contacts_names=sensors.contacts.names,
                    xfrc_names=sensors.xfrc.names,
                    muscles_names=sensors.muscles.names,
                    adhesions_names=sensors.adhesions.names,
                    visuals_names=sensors.visuals.names,
                ),
                network=network_log,
            )],
        )
        self._record_index = 0
        self._recording = True

    def _stop_recording(self):
        """Stop recording and save to file."""
        self._recording = False
        if self._record_data is not None:
            self._record_data.to_file(self._record_path, iteration=self._record_index)
            pylog.info("Recording saved to %s (%d iterations)", self._record_path, self._record_index)
            self._record_data = None

    def _record_step(self):
        """Copy current iteration's sensor and network data into the recording buffer."""
        if not self._recording:
            return
        # task.iteration was already incremented by task.after_step(),
        # so read from the previous index to match what was just simulated.
        sim_idx = (self.task.iteration - 1) % self.task.buffer_size
        rec_idx = self._record_index

        # Sensors
        sim_sensors = self.farms_data.sensors
        rec_sensors = self._record_data.animats[0].sensors
        for attr in ('links', 'joints', 'contacts', 'xfrc', 'muscles', 'adhesions', 'visuals'):
            src = getattr(sim_sensors, attr, None)
            dst = getattr(rec_sensors, attr, None)
            if src is not None and dst is not None:
                dst.array[rec_idx] = src.array[sim_idx]

        # Network
        if self._record_data.animats[0].network is not None:
            sim_log = self.network.log
            rec_log = self._record_data.animats[0].network
            net_idx = (self.task.iteration - 1) % sim_log.outputs.array.shape[0]
            rec_log.states.array[rec_idx] = sim_log.states.array[net_idx]
            rec_log.outputs.array[rec_idx] = sim_log.outputs.array[net_idx]
            rec_log.external_inputs.array[rec_idx] = sim_log.external_inputs.array[net_idx]

        self._record_index += 1
        if self._record_index >= self._record_duration:
            self._stop_recording()

    def _step(self, n):
        """Step the simulation by n iterations. Positive = forward, negative = rewind view."""
        if self.sim is None:
            return

        # Pause on manual step
        if self.playback_state == PlaybackState.PLAYING:
            self.playback_state = PlaybackState.PAUSED

        task = self.task
        if n > 0:
            for _ in range(n):
                if task.iteration >= task.n_iterations:
                    task.iteration = 0
                    task.sim_iteration = 0
                    self.sim._env._step_count = 0
                    self.sim._env._reset_next_step = False
                self.sim.update_step_options()
                for _ in range(task.cb_sub_steps):
                    if self._mujoco_win._initialized:
                        self._mujoco_win.apply_perturbation()
                    self.hooks["pre_substep"].fire(self, task.iteration, self.sim.physics)
                    self.sim._env.step(action=None)
                    self.hooks["post_substep"].fire(self, task.iteration, self.sim.physics)
                self._record_step()
        else:
            # Rewind view index (no physics rewind)
            new_iter = max(0, task.iteration + n)
            task.iteration = new_iter

    # Menu
    def menu(self):
        if imgui.begin_menu("FARMSIM"):
            if imgui.menu_item_simple("Open"):
                self.load_experiment()
            if imgui.menu_item_simple("Reload", enabled=self.sim is not None):
                self._reload_experiment()
            if imgui.menu_item_simple("Close", enabled=self.sim is not None):
                self._teardown()
            imgui.separator()
            if imgui.begin_menu("Add", enabled=self.sim is not None):
                if imgui.menu_item_simple("Plot Window"):
                    FARMSIMExtension._PLOT_WINDOW_COUNTER += 1
                    self._new_plot_name = f"Plot {FARMSIMExtension._PLOT_WINDOW_COUNTER}"
                    self._show_new_plot_popup = True
                imgui.end_menu()
            imgui.separator()
            if imgui.begin_menu("Windows"):
                for window in self.windows.values():
                    clicked, _ = imgui.menu_item(
                        window.name, "", window.visible, True,
                    )
                    if clicked:
                        window.toggle_visibility()
                imgui.end_menu()
            imgui.separator()
            if imgui.menu_item_simple("Reset Layout", enabled=self.sim is not None):
                self.on_reset_state()
            imgui.end_menu()

        # New plot window name popup
        if self._show_new_plot_popup:
            imgui.open_popup("##new_plot_name")
            self._show_new_plot_popup = False
        if imgui.begin_popup("##new_plot_name"):
            imgui.text("Plot window name:")
            imgui.set_keyboard_focus_here()
            confirmed, self._new_plot_name = imgui.input_text(
                "##name", self._new_plot_name,
                imgui.InputTextFlags_.enter_returns_true,
            )
            if confirmed and self._new_plot_name:
                self._add_plot_window(self._new_plot_name)
                imgui.close_current_popup()
            if imgui.is_key_pressed(imgui.Key.escape):
                imgui.close_current_popup()
            imgui.end_popup()

        # Record popup
        if self._show_record_popup:
            imgui.open_popup("##record_config")
            self._show_record_popup = False
        if imgui.begin_popup("##record_config"):
            imgui.text("Record Configuration")
            imgui.separator()

            changed, val = imgui.input_int("Duration (iterations)", self._record_duration, step=100)
            if changed:
                self._record_duration = max(1, val)

            imgui.text(f"Save to: {self._record_path}")
            imgui.same_line()
            if imgui.small_button("Browse"):
                result = pfd.save_file(
                    "Save recording", self._record_path, ["*.hdf5"],
                ).result()
                if result:
                    self._record_path = result

            imgui.spacing()
            if imgui.button("Start Recording"):
                self._start_recording()
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel"):
                imgui.close_current_popup()

            imgui.end_popup()

    # Experiment loading
    def load_experiment(self, path: str = None):
        """Load an experiment config and set up the simulation."""
        if path is None:
            result = pfd.open_file(
                "Experiment options",
                default_path="",
                filters=["*.yaml"],
            ).result()
            path = result[0]

        try:

            self._teardown()

            original_cwd = os.getcwd()
            os.chdir(os.path.dirname(path))
            sys.path.append(original_cwd)
            exp = ExperimentOptions.load(path)
            # TODO: This should be added to physics options
            exp.animats[0].mujoco = {
                "use_site": True,
                "use_muscles": True,
                "use_frc_trq_sensors": True,
            }
            exp.animats[0].name = "Model"
            os.chdir(original_cwd)

            self.sim = simulation_setup(experiment_options=exp,)
            self._experiment_path = path
            self._config_win.load_file(path)
            self.registry = build_registry(self.sim)
            pylog.info(f"Loaded experiment: {path}")

            # Restore saved plot windows, or create defaults
            if not self._restore_plot_windows():
                self._create_default_plot_windows()
            self.init_windows()

        except Exception as e:
            pylog.error(f"Failed to load experiment: {e}")
            console.print_exception(show_locals=True)
            self.sim = None

    def _reload_experiment(self):
        """Reload the current experiment from disk."""
        if hasattr(self, '_experiment_path'):
            self.load_experiment(self._experiment_path)

    def _create_default_plot_windows(self):
        """Create default plot windows. Connect to ``create_default_plots`` hook to override."""
        if self.hooks["create_default_plots"]:
            self.hooks["create_default_plots"].fire(self)
        else:
            self._add_plot_window()

    def _add_plot_window(self, name: str = None):
        """Create a new empty plot window the user can configure."""
        if name is None:
            FARMSIMExtension._PLOT_WINDOW_COUNTER += 1
            name = f"Plot {FARMSIMExtension._PLOT_WINDOW_COUNTER}"
        win = PlotWindow(self, PlotWindowConfig(name=name))
        self.register_window(win)
        win.initialize()
        if self._bottom_dock_id:
            from farms_app.core import layout
            layout.dock_window(win.window_id, self._bottom_dock_id)

    # Simulation stepping
    def on_update(self, dt):
        if self.sim is None or self.playback_state != PlaybackState.PLAYING:
            return

        now = time.perf_counter()
        wall_elapsed = now - self._last_update_time
        self._last_update_time = now

        sim_budget = min(
            (wall_elapsed + self._dt_remainder) * self.playback_speed, 0.1,
        )
        n_steps, self._dt_remainder = divmod(sim_budget, self.sim.task.timestep)
        n_steps = int(n_steps)

        task = self.task
        for _ in range(n_steps):
            if task.iteration >= task.n_iterations:
                task.iteration = 0
                task.sim_iteration = 0
                self.sim._env._step_count = 0
                self.sim._env._reset_next_step = False
            self.sim.update_step_options()
            for _ in range(task.cb_sub_steps):
                if self._mujoco_win._initialized:
                    self._mujoco_win.apply_perturbation()
                self.hooks["pre_substep"].fire(self, task.iteration, self.sim.physics)
                self.sim._env.step(action=None)
                self.hooks["post_substep"].fire(self, task.iteration, self.sim.physics)
            self._record_step()

    # Input & lifecycle
    def on_event(self):
        if self._mujoco_win._initialized:
            self._mujoco_win.handle_input()

    def _teardown(self):
        """Clean up the current simulation if one exists."""
        # Save plot configs before removing windows
        if self.sim is not None:
            self._last_saved_state = self.on_save_state()
            pylog.info("Tearing down current simulation")
            self.sim = None
        # Remove dynamic windows (keep persistent windows like the viewport)
        to_remove = [
            name for name, w in self.windows.items()
            if isinstance(w, PlotWindow)
        ]
        for name in to_remove:
            self.unregister_window(self.windows[name])
        self.registry = DataRegistry()
        self.playback_state = PlaybackState.STOPPED
        self._dt_remainder = 0.0
        self._view_offset = 0

    # State persistence
    def on_save_state(self) -> dict:
        """Save plot window configs so they persist across runs."""
        # If sim is torn down, windows are gone — return cached state
        plot_windows = [w for w in self.windows.values() if isinstance(w, PlotWindow)]
        if not plot_windows and self.sim is None:
            return getattr(self, '_last_saved_state', {})

        plot_configs = []
        for window in plot_windows:
            cfg = window.config
            plot_configs.append({
                "name": cfg.name,
                "layout": cfg.layout,
                "visible": window.visible,
                "plots": [
                    {
                        "x_source": p.x_source,
                        "y_sources": list(p.y_sources),
                        "title": p.title,
                        "y_label": p.y_label,
                        "axis_limits": {
                            "x": list(window._axis_limits[i]["x"]),
                            "y": list(window._axis_limits[i]["y"]),
                        } if i in window._axis_limits else None,
                        "signal_styles": {
                            name: {
                                "plot_type": s.plot_type,
                                "thickness": s.thickness,
                                "marker": s.marker,
                                "marker_size": s.marker_size,
                                "fill": s.fill,
                                "fill_alpha": s.fill_alpha,
                            }
                            for name, s in p.signal_styles.items()
                        } if p.signal_styles else None,
                    }
                    for i, p in enumerate(cfg.plots)
                ],
            })
        # Preserve hidden plot configs from previous sessions
        plot_configs.extend(getattr(self, '_hidden_plot_configs', []))

        return {
            "plot_windows": plot_configs,
            "experiment_path": getattr(self, '_experiment_path', None),
        }

    def on_restore_state(self, state: dict):
        """Restore plot windows from saved state."""
        self._saved_state = state

    def on_reset_state(self):
        """Reset to default plot windows, clearing saved and hidden state."""
        self._saved_state = None
        self._hidden_plot_configs = []
        # Remove existing plot windows
        to_remove = [name for name, w in self.windows.items() if isinstance(w, PlotWindow)]
        for name in to_remove:
            self.unregister_window(self.windows[name])
        # Recreate defaults
        if self.sim is not None:
            self._create_default_plot_windows()
            self.init_windows()

    def _restore_plot_windows(self):
        """Recreate plot windows from saved state. Called after experiment load."""
        state = getattr(self, '_saved_state', None)
        if state is None:
            return False

        # Only restore if same experiment
        saved_path = state.get("experiment_path")
        current_path = getattr(self, '_experiment_path', None)
        if saved_path != current_path:
            return False

        plot_configs = state.get("plot_windows", [])
        if not plot_configs:
            return False

        self._hidden_plot_configs = [
            cfg for cfg in plot_configs if not cfg.get("visible", True)
        ]
        for cfg_dict in plot_configs:
            if not cfg_dict.get("visible", True):
                continue
            plots = [
                PlotConfig(
                    x_source=p.get("x_source", "time"),
                    y_sources=p.get("y_sources", []),
                    title=p.get("title", ""),
                    y_label=p.get("y_label", ""),
                    axis_limits={
                        "x": tuple(al["x"]),
                        "y": tuple(al["y"]),
                    } if (al := p.get("axis_limits")) is not None else None,
                    signal_styles={
                        name: SignalStyle(**s)
                        for name, s in ss.items()
                    } if (ss := p.get("signal_styles")) else {},
                )
                for p in cfg_dict.get("plots", [])
            ]
            config = PlotWindowConfig(
                name=cfg_dict.get("name", "Plot"),
                layout=cfg_dict.get("layout", "subplots_vertical"),
                plots=plots,
            )
            win = PlotWindow(self, config)
            self.register_window(win)

        return True

    def cleanup(self):
        self._teardown()
