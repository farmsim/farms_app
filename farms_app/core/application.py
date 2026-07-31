""" Main script to run the FARMS app """

import time

from farms_app.backends.base import BaseBackend
from farms_app.backends.glfw_impl import OpenGLVersion
from farms_app.backends.manager import BackendManager
from farms_app.console import console
from farms_app.core.extension import ExtensionManager
from farms_app.core.fonts import load_fonts
from farms_app.core.inputs import InputManager
from farms_app.core.profiler import FrameTimer
from farms_core import pylog
from imgui_bundle import imgui

from .menus import render_main_menu
from .options import ApplicationOptions

_DEFAULT_OPTIONS_FILE = "options.yaml"


class FARMSApplication:
    """FARMS Application """

    def __init__(self, options: ApplicationOptions, options_path: str = None):
        """Initialization"""
        super().__init__()

        self._options: ApplicationOptions = options
        self._options_path = options_path or _DEFAULT_OPTIONS_FILE

        # Setup backend
        self.backend: BaseBackend = None
        self._io = None
        self._setup_backend(self._options)
        self.show_metrics_window = False

        self.fps_idle = options.fps_idle
        self.enable_idling = options.enable_idling
        self.is_idling = False

        # Fonts
        load_fonts(options.fonts)

        # Setup extensions
        self.extension_manager = ExtensionManager()
        self.extension_manager._saved_state = self._options.extension.state

        # Dockspace
        self.dockspace_id: int = 0

        # Quit confirmation
        self._quit_requested = False
        self._quit_confirmed = False
        self._save_on_quit = True

        # Screenshot: path set by menu after pfd dialog, forwarded to backend next frame
        self._screenshot_path: str | None = None

        # Experiment path to load on startup (set by CLI)
        self.experiment_path: str | None = None

        # Input
        self.input_manager: InputManager = InputManager()

        # Frame timer
        self.frame_timer = FrameTimer()
        self.extension_manager.frame_timer = self.frame_timer

    def _setup_backend(self, options: ApplicationOptions):
        """ Setup backend """
        backend_manager = BackendManager()
        self.backend = backend_manager.initialize(
            backend_type=options.backend.platform,
            gl_version=OpenGLVersion.GL2
        )
        self.backend.initialize(name=options.title)
        self._io = imgui.get_io()
        return backend_manager

    def fps_idling(self):
        """ Idle fps """

        self.is_idling = False
        if ((self.fps_idle > 0.0) and self.enable_idling):

            before_wait = time.time_ns()
            wait_timeout = 1.0 / self.fps_idle

            # Backend specific call that will wait for an event for a maximum duration of waitTimeout
            self.backend.event_timeout(wait_timeout)

            after_wait = time.time_ns()
            wait_duration = (after_wait - before_wait)
            wait_idle_expected = 1.0 / self.fps_idle
            self.is_idling = (wait_duration > wait_idle_expected * 0.9)

    @classmethod
    def from_options(cls, options: ApplicationOptions):
        """ Initialize using options """
        return cls(options)

    @staticmethod
    def load_options(path: str = _DEFAULT_OPTIONS_FILE) -> ApplicationOptions:
        """Load options from YAML. Falls back to defaults."""
        import os
        if os.path.exists(path):
            options = ApplicationOptions.load(path)
            pylog.info(f"Loaded options from {path}")
        else:
            options = ApplicationOptions.defaults()
        return options

    @classmethod
    def from_file(cls, path: str = _DEFAULT_OPTIONS_FILE):
        """Load options from YAML and initialize. Falls back to defaults."""
        return cls(cls.load_options(path), options_path=path)

    def render_menu(self):
        """ Render menu """
        render_main_menu(self)

    def run(self, on_ready=None):
        """Main run method.

        Args:
            on_ready: Optional callback ``f(app)`` invoked once on the first
                      frame, after extensions are enabled and the dockspace
                      is ready. Use this to register windows, connect hooks,
                      and load data — the dockspace_id is guaranteed valid.
        """

        _first = True
        _last_time = time.perf_counter()

        try:
            while not self.backend.should_close():

                # Frame timing
                now = time.perf_counter()
                dt = now - _last_time
                _last_time = now

                # Idling
                self.fps_idling()

                # Poll events
                self.backend.poll_events()

                # Deferred screenshot: arm backend one frame after request so
                # the menu UI is closed before the pixels are read
                if self._screenshot_path:
                    self.backend._screenshot_path = self._screenshot_path
                    self._screenshot_path = None

                # Pre-frame: raw GL rendering (before ImGui)
                self.extension_manager.pre_frame()

                # Start the Dear ImGui frame
                self.backend.begin_frame()
                self.dockspace_id = imgui.dock_space_over_viewport(
                    viewport=imgui.get_main_viewport()
                )

                # Debug
                if self.show_metrics_window:
                    self.show_metrics_window = imgui.show_metrics_window(self.show_metrics_window)

                # Render main menu
                self.render_menu()

                self.extension_manager.dockspace_id = self.dockspace_id

                if _first:
                    for ext_name in self._options.extension.auto_enable:
                        self.extension_manager.enable(ext_name)
                    # If experiment path is provided, load it in farmsim extension
                    if self.experiment_path:
                        farmsim_ext = self.extension_manager.get("farmsim")
                        farmsim_ext.load_experiment(self.experiment_path)
                    if on_ready:
                        on_ready(self)
                    _first = False

                # Global shortcuts (suppressed when modal open or typing)
                self.input_manager.process()

                # Tick all extensions: update(dt) -> event() -> render()
                self.frame_timer.begin_frame()
                self.extension_manager.tick(dt)
                self.frame_timer.end_frame()
                self.frame_timer.render_overlay()

                # Quit confirmation popup
                self._handle_quit_popup()

                # End the Dear ImGui frame
                self.backend.end_frame()
        except KeyboardInterrupt:
            pylog.info("Interrupted — saving state")
            self._save_on_quit = True
        finally:
            if self._save_on_quit:
                self._save_state()
            self.extension_manager.shutdown()
            self.backend.cleanup()

    def _save_state(self):
        """Save extension state into options and write to disk."""
        self._options.extension.auto_enable = list(self.extension_manager._enabled_exts.keys())
        self._options.extension.state = self.extension_manager.save_state()
        try:
            self._options.save(self._options_path)
            pylog.info(f"Saved options to {self._options_path}")
        except Exception as e:
            pylog.error(f"Error saving options: {e}")

    def _handle_quit_popup(self):
        """Intercept close requests and show a confirmation popup."""
        if self.backend.should_close() and not self._quit_confirmed:
            self.backend.cancel_close()
            self._quit_requested = True
            imgui.open_popup("Quit?")

        # Center the popup on the main viewport
        center = imgui.get_main_viewport().get_center()
        imgui.set_next_window_pos(center, imgui.Cond_.appearing, imgui.ImVec2(0.5, 0.5))

        visible, _ = imgui.begin_popup_modal("Quit?", flags=imgui.WindowFlags_.always_auto_resize)
        if not visible:
            return

        imgui.text("Save layout before quitting?")
        imgui.spacing()

        confirm = imgui.button("(S)ave & Quit") or imgui.is_key_pressed(imgui.Key.enter) or imgui.is_key_pressed(imgui.Key.s)
        if confirm:
            self._save_on_quit = True
            self._quit_confirmed = True
            imgui.close_current_popup()
            self.backend.request_close()
        imgui.same_line()
        if imgui.button("(Q)uit without saving") or imgui.is_key_pressed(imgui.Key.q):
            self._save_on_quit = False
            self._quit_confirmed = True
            imgui.close_current_popup()
            self.backend.request_close()
        imgui.same_line()
        cancel = imgui.button("(C)ancel") or imgui.is_key_pressed(imgui.Key.escape) or imgui.is_key_pressed(imgui.Key.c)
        if cancel:
            self._quit_requested = False
            imgui.close_current_popup()

        imgui.end_popup()
