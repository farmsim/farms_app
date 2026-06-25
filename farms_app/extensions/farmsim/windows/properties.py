""" Properties / inspector panel for MuJoCo simulation parameters """

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
from farms_app.core.window import Window
from imgui_bundle import imgui

if TYPE_CHECKING:
    from farms_app.extensions.farmsim.extension import FARMSIMExtension


# Enum display names
_INTEGRATORS = ["Euler", "RK4", "Implicit", "ImplicitFast"]
_CONES = ["Pyramidal", "Elliptic"]
_SOLVERS = ["PGS", "CG", "Newton"]
_JACOBIANS = ["Dense", "Sparse", "Auto"]


class PropertiesWindow(Window["FARMSIMExtension"]):
    """Live editing of MuJoCo parameters and simulation info."""

    def __init__(self, extension):
        super().__init__("Properties", extension)
        self._relay_names: list[str] = []
        self._relay_idx = 0

    @property
    def model(self):
        sim = self._extension.sim
        if sim is None:
            return None
        return sim.physics.model._model

    @property
    def data(self):
        sim = self._extension.sim
        if sim is None:
            return None
        return sim.physics.data._data

    def on_render(self):
        if self._extension.sim is None:
            self._relay_names.clear()
            imgui.text("No simulation loaded")
            return

        imgui.separator_text("Sim")
        self._render_sim_info()
        imgui.separator_text("Network")
        self._render_relay_control()
        self._render_weights()
        imgui.separator_text("Physics")
        self._render_physics_options()
        self._render_selected_body()

    # Simulation info
    def _render_sim_info(self):
        if imgui.collapsing_header("Simulation", imgui.TreeNodeFlags_.default_open):
            task = self._extension.task
            imgui.text(f"Iteration: {task.iteration} / {task.n_iterations}")
            imgui.text(f"Time: {task.iteration * task.timestep:.3f} s")
            imgui.text(f"Timestep: {task.timestep:.4f} s")
            imgui.text(f"Buffer: {task.buffer_size}")
            imgui.text(f"State: {self._extension.playback_state}")
            imgui.text(f"Speed: {self._extension.playback_speed}x")

    # Physics options
    def _render_physics_options(self):
        if not imgui.collapsing_header("Physics"):
            return

        model = self.model
        opt = model.opt

        # Integrator
        changed, idx = imgui.combo("Integrator", int(opt.integrator), _INTEGRATORS)
        if changed:
            opt.integrator = idx

        # Contact cone
        changed, idx = imgui.combo("Cone", int(opt.cone), _CONES)
        if changed:
            opt.cone = idx

        # Solver
        changed, idx = imgui.combo("Solver", int(opt.solver), _SOLVERS)
        if changed:
            opt.solver = idx

        # Jacobian
        changed, idx = imgui.combo("Jacobian", int(opt.jacobian), _JACOBIANS)
        if changed:
            opt.jacobian = idx

        imgui.separator()

        # Timestep
        changed, val = imgui.input_float("Timestep", opt.timestep, step=0.0001, format="%.5f")
        if changed:
            opt.timestep = max(1e-6, val)

        # Solver iterations
        changed, val = imgui.input_int("Iterations", opt.iterations)
        if changed:
            opt.iterations = max(1, val)

        # Noslip iterations
        changed, val = imgui.input_int("Noslip iter.", opt.noslip_iterations)
        if changed:
            opt.noslip_iterations = max(0, val)

        # Tolerance
        changed, val = imgui.input_float("Tolerance", opt.tolerance, step=0.0, format="%.2e")
        if changed:
            opt.tolerance = max(0.0, val)

        imgui.separator()

        # Gravity
        grav = list(opt.gravity)
        changed, grav = imgui.input_float3("Gravity", grav, format="%.2f")
        if changed:
            opt.gravity[:] = grav

    # Relay neuron control
    def _render_relay_control(self):
        network = self._extension.network
        if network is None:
            return

        if not imgui.collapsing_header("Drive", imgui.TreeNodeFlags_.default_open):
            return

        # Build relay list once per loaded network
        if not self._relay_names:
            self._relay_names = [
                node.name for node in network.options.nodes
                if node.model == "relay"
            ]
            self._relay_idx = 0

        if not self._relay_names:
            imgui.text_disabled("No relay neurons")
            return

        imgui.set_next_item_width(200)
        _, self._relay_idx = imgui.combo(
            "##relay_node", self._relay_idx, self._relay_names
        )

        name = self._relay_names[self._relay_idx]
        node_data = network.data.nodes[name]

        imgui.set_next_item_width(-1)
        _, node_data.external_input.values = imgui.drag_float(
            f"##{name}_ext_input",
            float(node_data.external_input.values),
            v_speed=0.05,
            v_min=0.0,
            v_max=1.5,
            format=f"{name}  %.3f",
        )

    # Network weights
    def _render_weights(self):
        network = self._extension.network
        if network is None:
            return

        if not imgui.collapsing_header("Weights"):
            return

        flags = (
            imgui.TableFlags_.borders
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.resizable
        )
        if imgui.begin_table("##weights", 3, flags):
            for col in ("Source", "Target", "Weight"):
                imgui.table_setup_column(col)
            imgui.table_headers_row()

            for row, edge in enumerate(network.data.edges):
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                imgui.text(edge.source)
                imgui.table_set_column_index(1)
                imgui.text(edge.target)
                imgui.table_set_column_index(2)
                imgui.push_id(row)
                w = float(edge.weight.values)
                _min, _max = (-10.0, 0.0) if w < 0.0 else (0.0, 10.0)
                _, edge.weight.values = imgui.drag_float(
                    "##w", w, v_speed=0.05, v_min=_min, v_max=_max,
                )
                imgui.pop_id()

            imgui.end_table()

    # Selected body info
    def _render_selected_body(self):
        if not imgui.collapsing_header("Selected Body"):
            return

        viewport = self._extension._mujoco_win
        if not viewport._initialized or viewport.mj_perturb is None:
            imgui.text_disabled("No viewport")
            return

        body_id = viewport.mj_perturb.select
        if body_id <= 0:
            imgui.text_disabled("No body selected (double-click in viewport)")
            return

        model = self.model
        data = self.data

        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        imgui.text(f"Name: {name or f'body_{body_id}'}")
        imgui.text(f"ID: {body_id}")
        imgui.separator()

        # Mass
        imgui.text(f"Mass: {model.body_mass[body_id]:.4f} kg")

        # Inertia
        inertia = model.body_inertia[body_id]
        imgui.text(f"Inertia: [{inertia[0]:.4e}, {inertia[1]:.4e}, {inertia[2]:.4e}]")

        # Position
        pos = data.xpos[body_id]
        imgui.text(f"Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

        # Velocity (if available)
        vel = data.cvel[body_id]
        imgui.text(f"Lin vel: [{vel[3]:.4f}, {vel[4]:.4f}, {vel[5]:.4f}]")
        imgui.text(f"Ang vel: [{vel[0]:.4f}, {vel[1]:.4f}, {vel[2]:.4f}]")
