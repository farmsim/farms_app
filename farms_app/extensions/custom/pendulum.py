import importlib
import os
import time
import traceback

import mujoco
import numpy as np
import OpenGL.GL as GL
from dm_control.rl.control import PhysicsError
from farms_app.backends.renderer.gl_framebuffer import MSAAFramebuffer
from farms_app.core.extension import Extension
from farms_app.core.widget import PlaybackState, SimulationToolbar
from farms_app.core.window import Window
from farms_app.utils import colors
from farms_core import pylog
from farms_core.experiment.options import ExperimentOptions
from farms_core.io import sdf
from farms_core.io.yaml import read_yaml
from farms_core.model.control import AnimatController, ControlType
from farms_core.model.data import AnimatData
from farms_core.model.options import (AnimatOptions, ArenaOptions,
                                      ControlOptions, JointOptions,
                                      LinkOptions, ModelOptions,
                                      MorphologyOptions, MotorOptions,
                                      MuscleOptions, SensorsOptions,
                                      SpawnLoader, SpawnOptions)
from farms_core.sensors.sensor_convention import sc
from farms_core.simulation.options import SimulationOptions, Simulator
from farms_muscle import rigid_tendon
from farms_network.core.network import Network
from farms_network.core.options import NetworkOptions
from farms_sim.simulation import run_simulation, simulation_setup
from imgui_bundle import imgui, imgui_ctx, implot, implot3d
from imgui_bundle import portable_file_dialogs as pfd


MJ_IMGUI_KEYMAP = {
    # special keys
    "/": imgui.Key.slash,
    "\\": imgui.Key.backslash,
    ",": imgui.Key.comma,
    ".": imgui.Key.period,
    ";": imgui.Key.semicolon,
    "'": imgui.Key.apostrophe,
    "[": imgui.Key.left_bracket,
    "]": imgui.Key.right_bracket,
    "-": imgui.Key.minus,
    "=": imgui.Key.equal,
    "`": imgui.Key.grave_accent,
    # letters A–Z
    **{chr(c): getattr(imgui.Key, chr(c).lower()) for c in range(ord("A"), ord("Z")+1)},
    **{str(i): getattr(imgui.Key, f"_{i}") for i in range(6)},
}


_mjGEOMSTRING = (
    (
        "Geom1", "1", "0",
        "Geom2", "1", "1",
        "Geom3", "1", "2",
        "Geom4", "0", "3",
        "Geom5", "0", "4",
        "Geom6", "0", "5",
    ),
)


def rotate(vector, theta):
    """Rotate vector"""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array(((cos_t, -sin_t), (sin_t, cos_t)))
    return np.dot(rotation, vector)


def direction(vector1, vector2):
    """Unit direction"""
    return (vector2-vector1)/np.linalg.norm(vector2-vector1)


def connect_positions(source, destination, dir_shift, perp_shift):
    """Connect positions"""
    connection_direction = direction(source, destination)
    connection_perp = rotate(connection_direction, 0.5*np.pi)
    new_source = (
        source
        + dir_shift*connection_direction
        + perp_shift*connection_perp
    )
    new_destination = (
        destination
        - dir_shift*connection_direction
        + perp_shift*connection_perp
    )
    return new_source, new_destination


def draw_table(network_options, network_data):
    """ Draw table """
    flags = (
        imgui.TableFlags_.borders | imgui.TableFlags_.row_bg | imgui.TableFlags_.resizable |
        imgui.TableFlags_.sortable
    )
    with imgui_ctx.begin("Controls"):
        edges = network_options.edges
        nodes = network_options.nodes
        n_edges = len(edges)
        if imgui.begin_table("Edges", 3, flags):
            weights = network_data.connectivity.weights
            for col in ("Source", "Target", "Weight"):
                imgui.table_setup_column(col)
            imgui.table_headers_row()
            for row, edge in enumerate(edges):
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                imgui.text(edge.source)
                imgui.table_set_column_index(1)
                imgui.text(edge.target)
                imgui.table_set_column_index(2)
                imgui.push_id(row)

                # Find where this edge's index lives in edge_indices
                weight_pos = np.where(np.array(network_data.connectivity.edge_indices) == row)[0][0]

                _min, _max = (-10.0, 10.0)
                if weights[weight_pos] < 0.0:
                    _min, _max = (-10.0, 0.0)
                else:
                    _min, _max = (0.0, 10.0)
                _, weights[weight_pos] = imgui.drag_float(
                    "##row",
                    float(weights[weight_pos]),
                    v_speed=0.05, v_min=_min, v_max=_max
                )
                imgui.pop_id()

            imgui.end_table()


def draw_play_pause_button(button_state):
    """ Draw button """

    button_title = "Pause" if button_state else "Play"
    with imgui_ctx.begin("Controls"):
        if imgui.button(button_title):
            button_state = not button_state
            print(button_state)
    return button_state


def col32(rgba):
    return imgui.color_convert_float4_to_u32(rgba)


class DiagramStyle:
    node_radius = 14
    node_shadow_offset = (4, 4)
    node_shadow_alpha = 0.22
    node_highlight_alpha = 0.30
    curve_thickness = 2.0
    arrow_size = 8.0
    node_border_thickness = 2.0

    # colors
    col_node_blue = (0.55, 0.72, 1.0, 1.0)
    col_node_red = (1.0, 0.65, 0.65, 1.0)
    col_border_dark = (0.15, 0.15, 0.25, 1.0)
    col_edge = col32((0.30, 0.30, 0.42, 0.55))
    col_shadow = (0.05, 0.05, 0.15)
    col_highlight = (1.0, 1.0, 1.0)
    col_activation = (1.0, 0.82, 0.25)

style = DiagramStyle()


class NetworkVisualizerWindow(Window):

    def __init__(self, extension, network: Network = None):
        name: str = "visualizer"
        super().__init__(name, extension)

    def on_initialize(self):
        """ On initialize """
        print("Network init")
        self.network = self._extension.network

        self.edges_p1p2 = [
          (
            self.network.options.nodes[self.network.options.nodes.index(edge.source)].visual['position'][:2],
            self.network.options.nodes[self.network.options.nodes.index(edge.target)].visual['position'][:2],
          )
          for edge in self.network.options.edges
        ]

        edges_xy = np.array(
            [
                self.network.options.nodes[node_idx].visual['position'][:2]
                for edge in self.network.options.edges
                for node_idx in (
                        self.network.options.nodes.index(edge.source),
                        self.network.options.nodes.index(edge.target),
                )
            ]
        )
        for index in range(len(edges_xy) - 1):
            edges_xy[index], edges_xy[index + 1] = connect_positions(
                edges_xy[index+1], edges_xy[index], 0.5, 0.0
            )
        self.edges_x = np.array(edges_xy[:, 0])
        self.edges_y = np.array(edges_xy[:, 1])

    def draw_node2(self, name, draw_list, pos, radius, color_fill):
        r = style.node_radius
        shadow_dx, shadow_dy = style.node_shadow_offset

        # Color conversion
        col_fill = color_fill
        col_border = col32(style.col_border_dark)
        col_shadow = col32((*style.col_shadow, style.node_shadow_alpha))
        col_highlight = col32((*style.col_highlight, style.node_highlight_alpha))

        if imgui.is_mouse_hovering_rect(
                imgui.ImVec2((pos[0] - radius, pos[1] - radius)),
                imgui.ImVec2((pos[0] + radius, pos[1] + radius))
        ):
            imgui.begin_tooltip()
            imgui.text(f"Neuron: {name}")
            imgui.text(f"Type: ....")
            # imgui.text_colored(f"Index: {i}", 0.8, 0.8, 0.3, 1)
            imgui.end_tooltip()

        # soft shadow
        draw_list.add_circle_filled(
            (pos[0] + shadow_dx, pos[1] + shadow_dy),
            r + 3,
            col_shadow
        )

        # fill
        draw_list.add_circle_filled(pos, r, col_fill)

        # border
        draw_list.add_circle(pos, r, col_border, 40, style.node_border_thickness)

        # internal highlight (upper-left)
        draw_list.add_circle_filled(
            (pos[0] - 3, pos[1] - 3),
            r * 0.55,
            col_highlight
        )

    def draw_node(self, name, draw_list, pos, radius, base_color, activation=0.0):
        r = radius
        ox, oy = style.node_shadow_offset
        act = float(np.clip(activation, 0.0, 1.0))

        # Gamma curve — makes small activations much more visually apparent
        act_visual = act ** 0.35  # was act (linear). 0.35 = aggressive, try 0.4-0.5 for subtler

        warm = [1.0, 0.82, 0.25]
        br, bg, bb = base_color[:3]
        fill_r = br + (warm[0] - br) * act_visual * 0.55
        fill_g = bg + (warm[1] - bg) * act_visual * 0.55
        fill_b = bb + (warm[2] - bb) * act_visual * 0.55

        col_fill      = col32((fill_r, fill_g, fill_b, act))
        col_border    = col32(style.col_border_dark)
        col_shadow    = col32((*style.col_shadow, style.node_shadow_alpha))
        col_highlight = col32((*style.col_highlight, style.node_highlight_alpha))
        col_glow      = col32((*style.col_activation, act * 0.18))

        # Hover tooltip
        hovered = imgui.is_mouse_hovering_rect(
            imgui.ImVec2((pos[0] - r, pos[1] - r)),
            imgui.ImVec2((pos[0] + r, pos[1] + r)),
        )
        if hovered:
            imgui.begin_tooltip()
            imgui.text(f"{name}")
            imgui.text(f"act: {act:.3f}")
            imgui.end_tooltip()

        # Glow — also gamma-corrected so it appears even at low activation
        if act > 0.1:  # lower threshold from 0.05
            for glow_r, glow_a in ((r + 9, 0.05), (r + 6, 0.09), (r + 3, 0.14)):
                draw_list.add_circle_filled(
                    pos, glow_r,
                    col32((*warm, glow_a * act_visual))
                )

        # Drop shadow (soft, two-pass)
        draw_list.add_circle_filled((pos[0] + ox + 1, pos[1] + oy + 1), r + 3, col32((*style.col_shadow, 0.10)))
        draw_list.add_circle_filled((pos[0] + ox,     pos[1] + oy),     r + 2, col_shadow)

        # Main fill
        draw_list.add_circle_filled(pos, r, col_fill)

        # Subtle inner rim (darker edge inside fill for depth)
        draw_list.add_circle(pos, r - 1.5, col32((0.0, 0.0, 0.0, 0.08)), 48, 2.5)

        # Border
        draw_list.add_circle(pos, r, col_border, 48, style.node_border_thickness)

        # Glint (upper-left specular highlight)
        draw_list.add_circle_filled(
            (pos[0] - r * 0.30, pos[1] - r * 0.30),
            r * 0.32,
            col32((*style.col_highlight, 0.45)),
        )
        # Smaller secondary glint
        draw_list.add_circle_filled(
            (pos[0] - r * 0.15, pos[1] - r * 0.44),
            r * 0.13,
            col32((*style.col_highlight, 0.25)),
        )

        # Label to the right of node, vertically centered
        label = name.split("_")[-1][:4]
        draw_list.add_text(
            (pos[0] + r + 5, pos[1] - 6),
            col32(style.col_border_dark),
            label,
        )


    def draw_bezier_connection(self, draw_list, color):
        for p1, p2 in self.edges_p1p2:
            p1 = implot.plot_to_pixels(p1)
            p2 = implot.plot_to_pixels(p2)

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            # Adaptive control points based on dominant direction
            if abs(dy) >= abs(dx):
                cp1 = (p1[0], p1[1] + dy * 0.45)
                cp2 = (p2[0], p2[1] - dy * 0.45)
            else:
                cp1 = (p1[0] + dx * 0.45, p1[1])
                cp2 = (p2[0] - dx * 0.45, p2[1])

            # Draw edge shadow first
            draw_list.add_bezier_cubic(
                p1, cp1, cp2, p2,
                col32((0.0, 0.0, 0.0, 0.12)),
                thickness=style.curve_thickness + 1.5,
            )
            # Then main edge
            draw_list.add_bezier_cubic(
                p1, cp1, cp2, p2,
                color,
                thickness=style.curve_thickness,
            )

    def on_update(self):
        """ On update of the application """

    def on_render(self):
        """ Render main extension dockspace """

        _, self.network.data.nodes['BS_input'].external_input.values = imgui.drag_float(
            "Drive",
            float(self.network.data.nodes['BS_input'].external_input.values),
            v_speed=0.05,
            v_min=0.0,
            v_max=1.5,
        )
        draw_table(self.network.options, self.network.data)
        self.draw_network()

    def draw_network(self):
        nodes = self.network.options.nodes
        edges = self.network.options.edges
        flags = (
            implot.AxisFlags_.no_label |
            implot.AxisFlags_.no_tick_labels |
            implot.AxisFlags_.no_tick_marks |
            implot.AxisFlags_.no_grid_lines
        )
        if implot.begin_plot(
                "vis", size=(-1, -1),
                flags=implot.Flags_.equal
        ):

            implot.setup_axis(implot.ImAxis_.x1, flags=flags)
            implot.setup_axis(implot.ImAxis_.y1, flags=flags)

            draw_list = implot.get_plot_draw_list()
            implot.push_plot_clip_rect()
            # safe to draw, nothing leaks outside plot
            self.draw_bezier_connection(draw_list, colors.COLORS['edge'])
            implot.pop_plot_clip_rect()

            for index, node in enumerate(nodes):
                p1 = implot.plot_to_pixels(node.visual['position'][:2])

                implot.push_plot_clip_rect()
                # safe to draw, nothing leaks outside plot
                self.draw_node(
                    node.name, draw_list, p1,
                    radius=style.node_radius,
                    base_color=node.visual['color'],
                    activation=self.network.data.outputs.array[index],
                )
                draw_list.add_text(p1, colors.COLORS['edge'], node.visual['label'].replace("\\textsubscript", "")[0])
                implot.pop_plot_clip_rect()

            implot.end_plot()


class AnalysisWindow(Window):
    """ FARMS Analysis Window """

    def __init__(
            self,
            extension,
            sim,
            farms_data,
    ):
        name: str = "Analysis"
        super().__init__(name, extension)
        self.sim = sim
        self.farms_data = farms_data
        self.network_data = extension.network.log
        # times for plots
        self.times = np.linspace(-1.0, 0.0, 1000)

        # Plot indexing
        self._network_plots_idx = {
            "RG_F": self.network_data.nodes._name_to_index["RG_F"],
            "RG_E": self.network_data.nodes._name_to_index["RG_E"],
            "In_F": self.network_data.nodes._name_to_index["RG_In_F"],
            "In_E": self.network_data.nodes._name_to_index["RG_In_E"],
            "Ia_F": self.network_data.nodes._name_to_index["motor_flexor_Ia"],
            "Ia_E": self.network_data.nodes._name_to_index["motor_extensor_Ia"],
            "Ib_F": self.network_data.nodes._name_to_index["motor_flexor_Ib"],
            "Ib_E": self.network_data.nodes._name_to_index["motor_extensor_Ib"],
            "II_F": self.network_data.nodes._name_to_index["motor_flexor_II"],
            "II_E": self.network_data.nodes._name_to_index["motor_extensor_II"],
        }

        # Force plots
        self._f_l_ce = np.linspace(0.2, 1.6, 50)
        self._f_active_force = np.zeros(np.shape(self._f_l_ce))
        self._f_passive_force = np.zeros(np.shape(self._f_l_ce))

        for step, _l_ce in enumerate(self._f_l_ce):
            self._f_active_force[step] = rigid_tendon.c_active_force(_l_ce, 0.0, 0.0)
            self._f_passive_force[step] = rigid_tendon.c_passive_force(_l_ce, 0.0, 0.0)

        self._f_v_ce = np.linspace(-2.0, 2.0, 50)
        self._f_force_velocity = np.zeros(np.shape(self._f_v_ce))

        for step, _v_ce in enumerate(self._f_v_ce):
            self._f_force_velocity[step] = rigid_tendon.c_force_velocity(_v_ce)

    def on_initialize(self):
        """ Initialize """

    def on_render(self):
        """ On render """
        buffer_iteration = (self.sim.task.iteration%self.sim.task.buffer_size)
        if imgui.begin_tab_bar("plots"):
            if imgui.begin_tab_item("Muscles")[0]:
                self.draw_muscle_force_plot(buffer_iteration, self.farms_data)
                imgui.end_tab_item()
            if imgui.begin_tab_item("Network")[0]:
                self.draw_network_plots(buffer_iteration, self.network_data)
                imgui.end_tab_item()
            if imgui.begin_tab_item("Dynamics")[0]:
                self.draw_joint_positions(buffer_iteration, self.farms_data)
                imgui.end_tab_item()
        imgui.end_tab_bar()
                # self.draw_feedbacks(buffer_iteration, self.farms_data)

    def draw_phase_markers(self):
        for xval in phase_markers:
            implot.plot_inf_lines(
                "##phase",
                np.array([xval]),
                flags=implot.InfLinesFlags_.vertical,
                spec=implot.Spec(line_color=(0.0, 0.0, 0.0, 0.8), line_weight=1.0),
            )


    def draw_network_plots(self, iteration, network_data):
        """ Draw Network Activity plots """

        _outputs = np.roll(network_data.outputs.array, -iteration, axis=0)

        panel_b_rows = [
            ("RG-F",  _outputs[:, 0], 0.0, 1.0),
            ("RG-E",  _outputs[:, 1], 0.0, 1.0),
            ("In-F",  _outputs[:, 2], 0.0, 1.0),
            ("In-E",  _outputs[:, 3], 0.0, 1.0),
            ("Ia-F",  _outputs[:, 4], -0.1, 4.0),
            ("II-F",  _outputs[:, 5], 0.0, 1.0),
            ("Ia-E",  _outputs[:, 6], -0.1, 4.0),
            ("Ib-E",  _outputs[:, 7], 0.0, 1.0),
        ]

        subplot_flags = (
            implot.Flags_.no_box_select |
            implot.Flags_.no_menus |
            implot.Flags_.no_legend
        )
        axis_flags_y = implot.AxisFlags_.auto_fit | implot.AxisFlags_.range_fit
        axis_flags_x = implot.AxisFlags_.no_tick_labels | implot.AxisFlags_.no_tick_marks
        axis_flags_x_last = implot.AxisFlags_.no_tick_marks  # show ticks only on bottom plot

        if implot.begin_subplots("Network", rows=8, cols=1, size=imgui.ImVec2((-1, -1)), flags=subplot_flags):
            for i, (label, idx) in enumerate(self._network_plots_idx.items()):
                is_last = (i == len(self._network_plots_idx) - 1)
                if implot.begin_plot(f"##panelB_{label}", flags=subplot_flags):
                    implot.setup_axes("", label)
                    implot.setup_axis(implot.ImAxis_.x1, "",
                                      flags=axis_flags_x_last if is_last else axis_flags_x)
                    implot.setup_axis(implot.ImAxis_.y1, label, flags=axis_flags_y)
                    implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
                    implot.plot_line(f"##{label}", self.times, np.ascontiguousarray(_outputs[:, idx]),
                                     spec=implot.Spec(line_color=(0.0, 0.0, 0.0, 1.0), line_weight=1.2))
                    implot.end_plot()
            implot.end_subplots()

    def draw_muscle_force_plot(self, iteration, farms_data):
        """ Draw muscle force plot """

        muscle_names = ['flexor', 'extensor']

        i_s = iteration - 1

        n = 100
        idx = np.array([(i_s - n + 1 + k) % self.sim.task.buffer_size for k in range(n)])

        muscle_colors = {
            'flexor':   (0.2, 0.4, 1.0),
            'extensor': (0.0, 0.7, 0.3),
        }

        l_opt = 1.0
        alpha_ramp = np.linspace(0.05, 0.5, n)
        plot_size = imgui.ImVec2(400, 300)

        # --- Collect data ---
        muscle_data = {}
        for muscle_name in muscle_names:
            mi = farms_data.sensors.muscles.names.index(muscle_name)
            activation     = np.array(farms_data.sensors.muscles.activations_all()[:, mi])
            active         = np.array(farms_data.sensors.muscles.active_forces_all()[:, mi])
            passive        = np.array(farms_data.sensors.muscles.passive_forces_all()[:, mi])
            fiber_length   = np.array(farms_data.sensors.muscles.fiber_lengths_all()[:, mi])
            forces         = np.array(farms_data.sensors.muscles.force_velocities_all()[:, mi])
            fiber_velocity = np.array(farms_data.sensors.muscles.fiber_velocities_all()[:, mi])
            muscle_data[muscle_name] = dict(
                activation=activation,
                fiber_length=fiber_length / l_opt,
                fiber_velocity=fiber_velocity,
                total_force=activation * active + passive,
                forces=forces,
            )

        # ── Row 1: Force-Length (flexor) | Force-Length (extensor) ───────────────────
        for i, muscle_name in enumerate(muscle_names):
            if i > 0:
                imgui.same_line()
            d = muscle_data[muscle_name]
            r, g, b = muscle_colors[muscle_name]

            if implot.begin_plot(f"Force-Length ({muscle_name})", size=plot_size,
                flags=implot.Flags_.no_box_select | implot.Flags_.no_menus,
            ):
                implot.setup_axes("l / l_opt", "Force (N)")
                implot.setup_axes_limits(0.0, 1.6, 0.0, 1.5, implot.Cond_.once)

                implot.plot_shaded(
                    "shortening##fl",
                    np.array([0.0, 1.0]), np.array([1.5, 1.5]), np.array([0.0, 0.0]),
                    spec=implot.Spec(fill_color=(0.4, 0.6, 1.0, 0.08)),
                )
                implot.plot_shaded(
                    "lengthening##fl",
                    np.array([1.0, 1.6]), np.array([1.5, 1.5]), np.array([0.0, 0.0]),
                    spec=implot.Spec(fill_color=(1.0, 0.5, 0.2, 0.08)),
                )
                implot.plot_line("active",  self._f_l_ce / l_opt, self._f_active_force,
                                 spec=implot.Spec(line_color=(1.0, 0.0, 0.0, 1.0), line_weight=1.5))
                implot.plot_line("passive", self._f_l_ce / l_opt, self._f_passive_force,
                                 spec=implot.Spec(line_color=(0.0, 1.0, 0.0, 1.0), line_weight=1.5))

                implot.plot_line(
                    f"{muscle_name}##fl_trail",
                    d['fiber_length'][idx], d['total_force'][idx],
                    spec=implot.Spec(line_color=(r, g, b, 0.5), line_weight=1.0),
                )
                implot.plot_scatter(
                    f"{muscle_name} (now)##fl",
                    d['fiber_length'][i_s:i_s+1], d['total_force'][i_s:i_s+1],
                    spec=implot.Spec(marker_size=6.0, marker_fill_color=(r, g, b, 1.0)),
                )

                implot.end_plot()

        # ── Row 2: Force-Velocity (flexor) | Force-Velocity (extensor) ───────────────
        for i, muscle_name in enumerate(muscle_names):
            if i > 0:
                imgui.same_line()
            d = muscle_data[muscle_name]
            r, g, b = muscle_colors[muscle_name]

            if implot.begin_plot(f"Force-Velocity ({muscle_name})", size=plot_size,
                flags=implot.Flags_.no_box_select | implot.Flags_.no_menus,
            ):
                implot.setup_axes("v_ce", "Force velocity factor")
                implot.setup_axes_limits(-2.0, 2.0, 0.0, 2.0, implot.Cond_.once)

                implot.plot_shaded(
                    "shortening##fv",
                    np.array([-2.0, 0.0]), np.array([2.0, 2.0]), np.array([0.0, 0.0]),
                    spec=implot.Spec(fill_color=(0.4, 0.6, 1.0, 0.08)),
                )
                implot.plot_shaded(
                    "lengthening##fv",
                    np.array([0.0, 2.0]), np.array([2.0, 2.0]), np.array([0.0, 0.0]),
                    spec=implot.Spec(fill_color=(1.0, 0.5, 0.2, 0.08)),
                )
                implot.plot_line("f-v curve", self._f_v_ce, self._f_force_velocity,
                                 spec=implot.Spec(line_color=(0.9, 0.9, 0.9, 1.0), line_weight=1.5))

                implot.plot_line(
                    f"{muscle_name}##fv_trail",
                    d['fiber_velocity'][idx], d['forces'][idx],
                    spec=implot.Spec(line_color=(r, g, b, 0.5), line_weight=1.0),
                )
                implot.plot_scatter(
                    f"{muscle_name} (now)##fv",
                    d['fiber_velocity'][i_s:i_s+1], d['forces'][i_s:i_s+1],
                    spec=implot.Spec(marker_size=6.0, marker_fill_color=(r, g, b, 1.0)),
                )

                implot.end_plot()

    def draw_joint_positions(self, iteration, farms_data):

        joint_names = farms_data.sensors.joints.names
        joints_data = np.roll(farms_data.sensors.joints.array, -iteration, axis=0)

        if implot.begin_subplots("##Dynamics", 2, 1, imgui.ImVec2(-1, -1),):
            if implot.begin_plot(f"Position"):
                implot.setup_axis(implot.ImAxis_.y1, "Angle [deg]")
                implot.setup_axis(
                    implot.ImAxis_.x1,
                    "Time", flags=(implot.AxisFlags_.no_tick_labels | implot.AxisFlags_.no_tick_marks)
                )
                implot.setup_axis_links(implot.ImAxis_.x1, implot.BoxedValue(-1.0), implot.BoxedValue(0.0))
                implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
                implot.plot_line(
                    joint_names[0],
                    self.times,
                    np.ascontiguousarray(np.rad2deg(joints_data[:, 0, sc.joint_position]))
                )
                implot.end_plot()
            if implot.begin_plot(f"Velocity"):
                implot.setup_axis(implot.ImAxis_.y1, "Angle [deg]")
                implot.setup_axis(
                    implot.ImAxis_.x1,
                    "Time", flags=(implot.AxisFlags_.no_tick_labels | implot.AxisFlags_.no_tick_marks)
                )
                implot.setup_axis_links(implot.ImAxis_.x1, implot.BoxedValue(-1.0), implot.BoxedValue(0.0))
                implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
                implot.plot_line(
                    joint_names[0],
                    self.times,
                    np.ascontiguousarray(np.rad2deg(joints_data[:, 0, sc.joint_velocity]))
                )
                implot.end_plot()
            if implot.begin_plot(f"Torque"):
                implot.setup_axis(implot.ImAxis_.y1, "Angle [deg]")
                implot.setup_axis(
                    implot.ImAxis_.x1,
                    "Time", flags=(implot.AxisFlags_.no_tick_labels | implot.AxisFlags_.no_tick_marks)
                )
                implot.setup_axis_links(implot.ImAxis_.x1, implot.BoxedValue(-1.0), implot.BoxedValue(0.0))
                implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
                implot.plot_line(
                    joint_names[0],
                    self.times,
                    np.ascontiguousarray(np.rad2deg(joints_data[:, 0, sc.joint_torque]))
                )
                implot.end_plot()
            implot.end_subplots()

            # implot.end_subplots()

    def draw_feedbacks(self, iteration, farms_data):

        muscle_names = farms_data.sensors.muscles.names
        muscles_data = farms_data.sensors.muscles

        if iteration < 1000:
            raw = np.array(muscles_data.Ia_feedbacks_all()[:iteration, :])
            if iteration == 0:
                plot_data = np.zeros((1000, raw.shape[1] if raw.ndim == 2 else len(muscle_names)))
            else:
                padding = np.zeros((1000 - iteration, raw.shape[1]))
                plot_data = np.vstack([padding, raw])
        else:
            plot_data = np.array(muscles_data.Ia_feedbacks_all()[iteration-1000:iteration, :])

        if implot.begin_plot(f"Feedbacks"):
            implot.setup_axis(implot.ImAxis_.y1, "")
            implot.setup_axis(
                implot.ImAxis_.x1,
                "", flags=(implot.AxisFlags_.no_tick_labels | implot.AxisFlags_.no_tick_marks)
            )
            implot.setup_axis_links(implot.ImAxis_.x1, implot.BoxedValue(-1.0), implot.BoxedValue(0.0))
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
            for index, name in enumerate(muscle_names):
                implot.plot_line(
                    muscle_names[index], self.times, np.array(muscles_data.Ia_feedbacks_all()[:, index])
                )
            implot.end_plot()

        if iteration < 1000:
            raw = np.array(muscles_data.activations_all()[:iteration, :])
            if iteration == 0:
                plot_data = np.zeros((1000, raw.shape[1] if raw.ndim == 2 else len(muscle_names)))
            else:
                padding = np.zeros((1000 - iteration, raw.shape[1]))
                plot_data = np.vstack([padding, raw])
        else:
            plot_data = np.array(muscles_data.activations_all()[iteration-1000:iteration, :])

        if implot.begin_plot(f"Activations"):
            implot.setup_axis(implot.ImAxis_.y1, "")
            implot.setup_axis(
                implot.ImAxis_.x1,
                "", flags=(implot.AxisFlags_.no_tick_labels | implot.AxisFlags_.no_tick_marks)
            )
            implot.setup_axis_links(implot.ImAxis_.x1, implot.BoxedValue(-1.0), implot.BoxedValue(0.0))
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
            for index, name in enumerate(muscle_names):
                implot.plot_line(
                    muscle_names[index], self.times, np.ascontiguousarray(plot_data[:, index])
                )
            implot.end_plot()
            # implot.end_subplots()


class MuJoCoWindow(Window):
    """ MuJoCo Window """

    def __init__(
            self,
            extension,
            model,
            data,
            window_size: tuple = (1280, 720)
    ):
        name: str = "MuJoCo"
        window_flags = (
            imgui.WindowFlags_.no_resize
        )
        super().__init__(name, extension, window_flags=window_flags)
        self._io = imgui.get_io()
        self.model = model
        self.data = data
        self.width, self.height = window_size[0], window_size[1]

        self.framebuffer = None
        self.depth_buffer = None
        self.texture_id = None

        # MjScene
        self.mj_camera = None
        self.mj_option = None
        self.mj_perturb = None
        self.mj_viewport = None
        self.mj_scene = None

        self.is_scene_hovered = False

    def on_initialize(self):
        """ On initializing the window """

        # Setup mujoco scene
        self.setup_mj_scene()

        # Create framebuffer
        self.fb = MSAAFramebuffer(self.width, self.height, samples=4)

    def render_main_scene(self):

        self._viewer_pos = imgui.get_cursor_screen_pos()

        self.fb.bind()

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.mj_option,
            self.mj_perturb,
            self.mj_camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.mj_scene
        )
        mujoco.mjr_render(self.mj_viewport, self.mj_scene, self.mj_context)

        self.fb.unbind()
        self.fb.resolve()

        avail_width, avail_height = imgui.get_content_region_avail()
        if avail_width <= 0 or avail_height <= 0:
            return

        target_aspect = self.width / self.height
        current_aspect = avail_width / avail_height
        if current_aspect > target_aspect:
            draw_height = avail_height
            draw_width = target_aspect * draw_height
        else:
            draw_width = avail_width
            draw_height = draw_width / target_aspect

        imgui.image(
            imgui.ImTextureRef(self.fb.texture_id),
            imgui.ImVec2((draw_width, draw_height)),
            uv0=imgui.ImVec2((1, 1)),
            uv1=imgui.ImVec2((0, 0)),
        )
        self._viewer_size = imgui.get_item_rect_size()
        self.is_scene_hovered = imgui.is_item_hovered()

    def on_render(self):
        """ Render window """
        self.render_main_scene()
        imgui.text(f"Time: {self._extension.sim.task.iteration}")

    def setup_mj_scene(self):
        self.mj_camera = mujoco.MjvCamera()
        self.mj_option = mujoco.MjvOption()
        self.mj_option.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = True
        self.model.vis.headlight.ambient[:] = [0.6]*3
        self.model.vis.headlight.diffuse[:] = [0.4]*3
        self.model.vis.headlight.specular[:] = [0.5]*3
        self.mj_perturb = mujoco.MjvPerturb()
        mujoco.mjv_defaultCamera(self.mj_camera)
        mujoco.mjv_defaultPerturb(self.mj_perturb)
        mujoco.mjv_defaultOption(self.mj_option)
        self.mj_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.mj_context)  # <-- added
        self.mj_scene = mujoco.MjvScene(self.model, maxgeom=1000000)
        self.mj_viewport = mujoco.MjrRect(0, 0, 0, 0)
        self.mj_viewport.width, self.mj_viewport.height = self.width, self.height
        self.mj_scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
        self.mj_scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        self.mj_scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        # self.mj_option.flags[mujoco.mjtVisFlag.mjVIS_SKYBOX] = True

    def run_simulation(self):
        """ Run Simulation """
        self.start_time = self.data.time
        while (self.data.time - self.start_time < 1.0/60.0):
            mujoco.mj_step(self.model, self.data)

        if imgui.is_item_hovered():
            self.mouse_interactions()
            self.keyboard_interactions()

    def __mj_keys(self, mjSTRING: tuple[str, str, str], mj_flags):
        for j, _opt in enumerate(mjSTRING):
            key_str = _opt[2]
            if not key_str:
                continue        # Skip if no key assigned
            key_enum = MJ_IMGUI_KEYMAP.get(key_str)
            if key_enum is None:
                continue
            if imgui.is_key_pressed(key_enum):
                mj_flags[j] = not mj_flags[j]

    def keyboard_interactions(self):
        """ keyboard interactions """
        self.__mj_keys(mujoco.mjRNDSTRING, self.mj_scene.flags)
        self.__mj_keys(mujoco.mjVISSTRING, self.mj_option.flags)
        self.__mj_keys(_mjGEOMSTRING, self.mj_option.geomgroup)

    def _select_body(self, gl_x, gl_y):
        """Raycast into scene and select body under cursor."""
        aspect = self.width / self.height
        geom_id = np.array([-1], dtype=np.int32)
        flex_id = np.array([-1], dtype=np.int32)
        skin_id = np.array([-1], dtype=np.int32)
        sel_pos = np.zeros(3, dtype=np.float64)

        body_id = mujoco.mjv_select(
            self.model,
            self.data,
            self.mj_option,
            aspect,
            gl_x / self.width,
            gl_y / self.height,
            self.mj_scene,
            sel_pos,
            geom_id,
            flex_id,
            skin_id,
        )
        return body_id, sel_pos

    def _screen_to_mujoco(self, mouse_pos):
        """Convert imgui screen coords to MuJoCo viewport coords."""
        viewer_pos = self._viewer_pos
        rel_x = mouse_pos.x - viewer_pos.x
        rel_y = mouse_pos.y - viewer_pos.y
        gl_x = int(rel_x * (self.width / self._viewer_size.x))
        gl_y = int(self.height - rel_y * (self.height / self._viewer_size.y))
        return gl_x, gl_y

    def mouse_interactions(self):
        """Mouse interactions matching native MuJoCo viewer behavior."""
        if not self.is_scene_hovered:
            # Clear perturbation if mouse leaves scene
            if self.mj_perturb.active != 0:
                self.mj_perturb.active = 0
            return

        mouse_pos = self._io.mouse_pos
        mouse_delta = self._io.mouse_delta
        mouse_wheel = self._io.mouse_wheel
        ctrl_held = self._io.key_ctrl
        shift_held = self._io.key_shift

        # --- Double-click to select/deselect body ---
        if imgui.is_mouse_double_clicked(imgui.MouseButton_.left):
            gl_x, gl_y = self._screen_to_mujoco(mouse_pos)
            body_id, sel_pos = self._select_body(gl_x, gl_y)

            if body_id > 0:
                self.mj_perturb.select = body_id
                self.mj_perturb.selectpos = sel_pos
                mujoco.mjv_initPerturb(self.model, self.data, self.mj_scene, self.mj_perturb)
            else:
                # Double-clicked empty space — deselect
                self.mj_perturb.select = 0
                self.mj_perturb.active = 0

        # --- Perturbation drag (requires a selected body) ---
        if ctrl_held and self.mj_perturb.select > 0:
            if imgui.is_key_down(imgui.Key.mouse_left):
                if shift_held:
                    # Ctrl + Shift + drag = rotate (torque)
                    self.mj_perturb.active = mujoco.mjtPertBit.mjPERT_ROTATE
                    mujoco.mjv_movePerturb(
                        self.model, self.data,
                        mujoco.mjtMouse.mjMOUSE_ROTATE_H,
                        mouse_delta.x / self.width, 0.0,
                        self.mj_scene, self.mj_perturb,
                    )
                    mujoco.mjv_movePerturb(
                        self.model, self.data,
                        mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                        0.0, -mouse_delta.y / self.height,
                        self.mj_scene, self.mj_perturb,
                    )
                else:
                    # Ctrl + drag = translate (force)
                    self.mj_perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
                    mujoco.mjv_movePerturb(
                        self.model, self.data,
                        mujoco.mjtMouse.mjMOUSE_MOVE_H,
                        mouse_delta.x / self.width, 0.0,
                        self.mj_scene, self.mj_perturb,
                    )
                    mujoco.mjv_movePerturb(
                        self.model, self.data,
                        mujoco.mjtMouse.mjMOUSE_MOVE_V,
                        0.0, -mouse_delta.y / self.height,
                        self.mj_scene, self.mj_perturb,
                    )

                mujoco.mjv_applyPerturbPose(self.model, self.data, self.mj_perturb, 0)
                mujoco.mjv_applyPerturbForce(self.model, self.data, self.mj_perturb)
            else:
                # Ctrl held but not dragging — clear active but keep selection
                self.mj_perturb.active = 0

        elif not ctrl_held:
            # Clear active perturbation when ctrl released, keep selection
            self.mj_perturb.active = 0

            # --- Camera controls ---
            if imgui.is_key_down(imgui.Key.mouse_left):
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_H,
                    -mouse_delta.x / self.width, 0.0, self.mj_scene, self.mj_camera)
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                    0.0, mouse_delta.y / self.height, self.mj_scene, self.mj_camera)
            elif imgui.is_key_down(imgui.Key.mouse_right):
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_H,
                    -mouse_delta.x / self.width, 0.0, self.mj_scene, self.mj_camera)
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_V,
                    0.0, mouse_delta.y / self.height, self.mj_scene, self.mj_camera)

        # --- Scroll to zoom (always active) ---
        if imgui.is_key_down(imgui.Key.mouse_wheel_y):
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                0.0, np.sign(mouse_wheel) * 0.05, self.mj_scene, self.mj_camera)

    def _mouse_interactions(self):
        """ Mouse interactions """
        mouse_pos = self._io.mouse_pos
        mouse_delta = self._io.mouse_delta
        mouse_wheel = self._io.mouse_wheel
        if imgui.is_key_down(imgui.Key.mouse_left):
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_ROTATE_H,
                -mouse_delta.x / self.width,
                0.0,
                self.mj_scene,
                self.mj_camera,
            )
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                0.0,
                mouse_delta.y / self.height,
                self.mj_scene,
                self.mj_camera,
            )
        elif imgui.is_key_down(imgui.Key.mouse_right):
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_MOVE_H,
                -mouse_delta.x / self.width,
                0.0,
                self.mj_scene,
                self.mj_camera,
            )
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_MOVE_V,
                0.0,
                mouse_delta.y / self.height,
                self.mj_scene,
                self.mj_camera,
            )
        elif imgui.is_key_down(imgui.Key.mouse_wheel_y):
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_ZOOM,
                0.0,
                np.sign(mouse_wheel)*0.05*1,
                self.mj_scene,
                self.mj_camera,
            )


class PendulumExtension(Extension):
    """ FARMS Simulation """

    def __init__(self):
        self.hide = False
        super().__init__(name="FARMS")
        self._io = imgui.get_io()
        self.sim = None
        self.dt_remainder = 0.0
        self.playback_speed = 0.1
        self._last_update_time = time.perf_counter()

        # FARMS
        self.exp_options = None
        self.simulation_options = None
        self.arena_options = None
        self.animat_options = None

        self.toolbar = SimulationToolbar()
        self.toolbar.on_play            = self._dummy
        self.toolbar.on_pause           = self._dummy
        self.toolbar.on_stop            = self._dummy
        self.toolbar.on_step_back       = self._dummy
        self.toolbar.on_step_back_many  = self._dummy
        self.toolbar.on_step_fwd        = self._dummy
        self.toolbar.on_step_fwd_many   = self._dummy
        self.toolbar.on_record          = self._dummy
        self.toolbar.on_speed_change    = self._dummy

    def _dummy(self):
        pass

    def register_windows(self):
        """ Register windows """
        self.register_window(MuJoCoWindow(self, self.model, self.data))
        self.register_window(AnalysisWindow(self, self.sim, self.farms_data))
        self.register_window(NetworkVisualizerWindow(self))
        # self.register_window(PropertiesWindow(self, self.sim, self.farms_data))

    def setup_simulation(self):

        # initialize data from options
        # self.animat_data = AnimatData.from_options(
        #     animat_options=self.exp_options.animats[0],
        #     simulation_options=self.exp_options.simulation,
        # )

        # Simulation
        self.sim = simulation_setup(
            experiment_options=self.exp_options,
            simulator=Simulator.MUJOCO,
        )

        # sim: Union[MuJoCoSimulation, None] = run_simulation(
        #     experiment_options=self.exp_options,
        #     simulator=Simulator.MUJOCO,
        # )

        self.model = self.sim.physics.model._model
        self.data = self.sim.physics.data._data

        # data: FARMS data for physics simulations
        self.farms_data: AnimatData = self.sim.task.data.animats[0]

        # network
        self.network: Network = self.sim.task.extensions[0].network

    def cleanup(self):
        """ Cleanup  """
        self.sim.end_extensions()

    def get_dependencies(self):
        return []

    def menu(self):
        """ Extension menu in main menu bar """
        if imgui.begin_menu("FARMS"):
            if imgui.menu_item_simple("Open"):
                try:
                    self.load_experiment()
                except Exception as e:
                    pylog.error(f"Unable to load the model {e}")
            if imgui.menu_item_simple("Reload"):
                pass
            if imgui.menu_item_simple("Close"):
                pass
            imgui.separator()
            if imgui.begin_menu("Simulation"):
                if imgui.menu_item_simple("Experiment"):
                    pass
                imgui.end_menu()
            if imgui.begin_menu("Windows"):
                for window in self.windows.values():
                    if imgui.menu_item_simple(window.name, selected=window.visible):
                        window.visible = not window.visible
                imgui.end_menu()
            imgui.end_menu()

    def on_update(self, dt):
        # Normal GUI operation
        if self.sim:
            # dt = 1/120
            # self.n_steps = int(1000 * (dt + self.dt_remainder) * self.playback_speed)
            # self.dt_remainder = (1000 * (dt + self.dt_remainder) * self.playback_speed - self.n_steps) / 1000

            # self.n_steps = max(0, self.n_steps)  # Ensure non-negative

            now = time.perf_counter()
            wall_elapsed = now - self._last_update_time
            self._last_update_time = now

            # Total simulation time to consume this frame (with remainder carried over)
            sim_budget = min((wall_elapsed + self.dt_remainder) * self.playback_speed, 0.1)  # max 100ms catchup

            self.n_steps, self.dt_remainder = divmod(sim_budget, self.sim.task.timestep)
            self.n_steps = int(self.n_steps)

            # Update GUI states

            for ii in range(self.n_steps):
                try:
                    if self.sim.task.iteration == (self.sim.task.n_iterations):
                        self.sim._env._step_count = 0
                        self.sim.task.iteration = 0
                        self.sim.task.sim_iteration = 0
                        self.sim._env._reset_next_step = False
                    self.sim.update_step_options()
                    for _ in range(self.sim.task.cb_sub_steps):
                        self.sim._env.step(action=None)
                except PhysicsError as err:
                    pylog.error(traceback.format_exc())
                    if self.sim.handle_exceptions:
                        return
                    raise err


    def load_experiment(self):
        """ Load experiment """
        # self.result = pfd.open_file(
        #     "Experiment options",
        #     default_path="",
        #     filters=("*.yaml",)
        # ).result()
        self.result = ["/Users/tatarama/projects/work/research/neuromechanics/misc/pendulum-fb/config/experiment.yaml",]
        os.chdir(os.path.dirname(self.result[0]))
        self.exp_options = ExperimentOptions.load(self.result[0])
        self.exp_options.animats[0].mujoco = {
            "use_site": True,
            "use_muscles": True,
            "use_frc_trq_sensors": True,
        }
        self.exp_options.animats[0].name = "Arm"

    def on_render(self):
        """ On render """

        self.toolbar.render(
            playback_state=PlaybackState.STOPPED,
            current_time=0.0,
            total_time=10.0,
            speed=1.0,
        )

        if self.exp_options:
            self.setup_simulation()
            self.register_windows()
            self.init_windows()
            self.exp_options = None

    def on_event(self):
        """ On events """
        mujoco_win = self.windows.get("MuJoCo")
        if mujoco_win and mujoco_win._initialized:
            if mujoco_win.is_scene_hovered:
                mujoco_win.mouse_interactions()
                mujoco_win.keyboard_interactions()
