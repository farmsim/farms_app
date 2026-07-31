""" Network visualizer: interactive topology diagram with activation coloring """

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from farms_app.core.window import Window
from imgui_bundle import imgui, implot, portable_file_dialogs as pfd

if TYPE_CHECKING:
    from farms_app.extensions.farmsim.extension import FARMSIMExtension


def _col32(rgba):
    return imgui.color_convert_float4_to_u32(rgba)


# Visual style
class _Style:
    node_radius = 14
    node_shadow_offset = (2, 2)
    node_shadow_alpha = 0.25
    curve_thickness = 1.5
    curve_alpha = 0.45
    arrow_length = 8.0
    arrow_half_width = 3.5
    node_border_thickness = 1.5
    label_offset_y = 4
    label_max_width = 80

    col_activation = (1.0, 0.82, 0.25)
    col_excitatory = (0.35, 0.65, 0.95)
    col_inhibitory = (0.95, 0.35, 0.35)

_style = _Style()


def _theme_colors():
    """Read colors from the current ImGui theme so the diagram adapts."""
    style = imgui.get_style()
    text = style.color_(imgui.Col_.text)
    bg = style.color_(imgui.Col_.window_bg)
    border = style.color_(imgui.Col_.border)
    is_dark = (bg.x * 0.299 + bg.y * 0.587 + bg.z * 0.114) < 0.5
    return {
        "border": (border.x, border.y, border.z, 0.8),
        "label": (text.x, text.y, text.z, 0.7),
        "shadow": (0.0, 0.0, 0.0) if is_dark else (0.05, 0.05, 0.15),
        "highlight": (1.0, 1.0, 1.0) if is_dark else (1.0, 1.0, 1.0),
    }


def _abbreviate(name):
    """Shorten a node name for display. Keep it readable but compact."""
    # Replace common prefixes
    name = name.replace("motor_", "m.")
    name = name.replace("extensor_", "ext.")
    name = name.replace("flexor_", "flx.")
    name = name.replace("RG_", "RG.")
    name = name.replace("BS_", "BS.")
    name = name.replace("In_", "In.")
    return name


# Window
class NetworkVisualizerWindow(Window["FARMSIMExtension"]):
    """Interactive network topology diagram with live activation coloring."""

    def __init__(self, extension):
        super().__init__(
            "Network Visualizer", extension,
            window_flags=imgui.WindowFlags_.menu_bar,
        )
        self._edges_p1p2 = []
        self._edge_weights = []
        self._network_ready = False
        self._needs_fit = False
        self._show_names = True
        self._tikz_neuron_shading = "ball"
        # Optional callable(raw_output: float) -> float in [0, 1].
        # Set this from app.py to remap model-specific outputs (e.g. phase → cos).
        self.activation_transform = None

    @property
    def network(self):
        return self._extension.network

    def _setup_network(self):
        """Build edge/node caches from the current network. Called lazily."""
        net = self.network

        self._edges_p1p2 = []
        self._edge_weights = []
        for i, edge in enumerate(net.options.edges):
            src_node = net.options.nodes[net.options.nodes.index(edge.source)]
            tgt_node = net.options.nodes[net.options.nodes.index(edge.target)]
            p1 = src_node.visual['position'][:2]
            p2 = tgt_node.visual['position'][:2]
            self._edges_p1p2.append((p1, p2))
            self._edge_weights.append(net.data.edges[i].weight)

        # Compute bounding box for initial fit
        positions = [n.visual['position'][:2] for n in net.options.nodes]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        pad = 0.5
        self._bounds = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

        self._network_ready = True
        self._needs_fit = True

    def _export_tikz(self):
        result = pfd.save_file("Export TikZ", "network.tex", ["*.tex"]).result()
        if result:
            from farms_app.extensions.farmsim.network_export import export_tikz
            export_tikz(self.network, result, neuron_shading=self._tikz_neuron_shading)

    def _export_matplotlib(self):
        result = pfd.save_file("Export Figure", "network.png", ["*.png", "*.pdf", "*.svg"]).result()
        if result:
            from farms_app.extensions.farmsim.network_export import export_matplotlib
            export_matplotlib(self.network, result)

    def on_render(self):
        if imgui.begin_menu_bar():
            if imgui.begin_menu("View"):
                _, self._show_names = imgui.menu_item("Show names", "", self._show_names)
                imgui.end_menu()
            if imgui.begin_menu("Export", enabled=self.network is not None):
                if imgui.menu_item_simple("TikZ (.tex)"):
                    self._export_tikz()
                if imgui.begin_menu("Neuron Shading"):
                    for shade in ("ball", "flat"):
                        if imgui.menu_item(shade, "", self._tikz_neuron_shading == shade)[0]:
                            self._tikz_neuron_shading = shade
                    imgui.end_menu()
                if imgui.menu_item_simple("Figure (.png)"):
                    self._export_matplotlib()
                imgui.end_menu()
            imgui.end_menu_bar()

        if self.network is None:
            self._network_ready = False
            imgui.text("No network loaded")
            return

        if not self._network_ready:
            self._setup_network()

        self._draw_diagram()

    # Diagram
    def _draw_diagram(self):
        net = self.network
        nodes = net.options.nodes
        axis_flags = (
            implot.AxisFlags_.no_label |
            implot.AxisFlags_.no_tick_labels |
            implot.AxisFlags_.no_tick_marks |
            implot.AxisFlags_.no_grid_lines
        )
        if implot.begin_plot("##network_vis", size=(-1, -1), flags=implot.Flags_.equal):
            implot.setup_axis(implot.ImAxis_.x1, flags=axis_flags)
            implot.setup_axis(implot.ImAxis_.y1, flags=axis_flags)
            if self._needs_fit:
                x_min, x_max, y_min, y_max = self._bounds
                implot.setup_axis_limits(implot.ImAxis_.x1, x_min, x_max)
                implot.setup_axis_limits(implot.ImAxis_.y1, y_min, y_max)
                self._needs_fit = False

            draw_list = implot.get_plot_draw_list()

            # Edges (behind nodes)
            implot.push_plot_clip_rect()
            self._draw_edges(draw_list)
            implot.pop_plot_clip_rect()

            # Nodes — collect pixel positions for label overlap avoidance
            node_positions = []
            view_iter = self._extension.view_iteration
            if self._extension._view_offset != 0:
                buf_idx = view_iter % net.buffer_size
                outputs = net.log.outputs.array[buf_idx]
            else:
                outputs = net.data.outputs.array
            for index, node in enumerate(nodes):
                pos = implot.plot_to_pixels(node.visual['position'][:2])
                raw = float(outputs[index]) if index < len(outputs) else 0.0
                activation = self.activation_transform(raw) if self.activation_transform is not None else raw

                implot.push_plot_clip_rect()
                self._draw_node(
                    draw_list, node.name, pos,
                    radius=_style.node_radius,
                    base_color=node.visual['color'],
                    activation=activation,
                )
                implot.pop_plot_clip_rect()
                node_positions.append(pos)

            # Labels (separate pass so we can skip overlaps)
            if self._show_names:
                implot.push_plot_clip_rect()
                self._draw_labels(draw_list, nodes, node_positions)
                implot.pop_plot_clip_rect()

            implot.end_plot()

    def _draw_labels(self, draw_list, nodes, positions):
        """Draw node labels below nodes, skipping any that would overlap."""
        theme = _theme_colors()
        col = _col32(theme["label"])
        r = _style.node_radius
        placed = []  # list of (x, y, w, h) rects for overlap checking

        for node, pos in zip(nodes, positions):
            label = _abbreviate(node.name)
            text_size = imgui.calc_text_size(label)

            # Truncate if too wide
            if text_size.x > _style.label_max_width:
                while len(label) > 3 and imgui.calc_text_size(label + "..").x > _style.label_max_width:
                    label = label[:-1]
                label = label + ".."
                text_size = imgui.calc_text_size(label)

            lx = pos[0] - text_size.x * 0.5
            ly = pos[1] + r + _style.label_offset_y
            lw = text_size.x
            lh = text_size.y

            # Check overlap with already placed labels
            overlaps = False
            for px, py, pw, ph in placed:
                if (lx < px + pw and lx + lw > px and
                    ly < py + ph and ly + lh > py):
                    overlaps = True
                    break

            if not overlaps:
                draw_list.add_text((lx, ly), col, label)
                placed.append((lx, ly, lw, lh))

    # Edge drawing
    def _draw_edges(self, draw_list):
        theme = _theme_colors()
        for i, (p1_plot, p2_plot) in enumerate(self._edges_p1p2):
            p1 = implot.plot_to_pixels(p1_plot)
            p2 = implot.plot_to_pixels(p2_plot)

            w = float(self._edge_weights[i].values)
            if w >= 0:
                base = _style.col_excitatory
            else:
                base = _style.col_inhibitory

            mag = min(abs(w), 5.0)
            thickness = _style.curve_thickness + mag * 0.3

            alpha = _style.curve_alpha
            edge_col = _col32((*base, alpha))
            shadow_col = _col32((*theme["shadow"], 0.1))

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1.0:
                continue

            r = _style.node_radius
            nx, ny = dx / dist, dy / dist
            p1_edge = (p1[0] + nx * r, p1[1] + ny * r)
            p2_edge = (p2[0] - nx * r, p2[1] - ny * r)

            edx = p2_edge[0] - p1_edge[0]
            edy = p2_edge[1] - p1_edge[1]

            # Add perpendicular offset for edges between the same pair
            # to reduce overlap in dense areas
            perp_x, perp_y = -ny, nx
            offset = (i % 3 - 1) * 4.0  # slight spread: -4, 0, +4
            p1_edge = (p1_edge[0] + perp_x * offset, p1_edge[1] + perp_y * offset)
            p2_edge = (p2_edge[0] + perp_x * offset, p2_edge[1] + perp_y * offset)

            edx = p2_edge[0] - p1_edge[0]
            edy = p2_edge[1] - p1_edge[1]

            if abs(edy) >= abs(edx):
                cp1 = (p1_edge[0], p1_edge[1] + edy * 0.4)
                cp2 = (p2_edge[0], p2_edge[1] - edy * 0.4)
            else:
                cp1 = (p1_edge[0] + edx * 0.4, p1_edge[1])
                cp2 = (p2_edge[0] - edx * 0.4, p2_edge[1])

            draw_list.add_bezier_cubic(
                p1_edge, cp1, cp2, p2_edge, shadow_col,
                thickness=thickness + 1.0,
            )
            draw_list.add_bezier_cubic(
                p1_edge, cp1, cp2, p2_edge, edge_col,
                thickness=thickness,
            )

            self._draw_arrowhead(draw_list, cp2, p2_edge, edge_col)

    def _draw_arrowhead(self, draw_list, cp_before, tip, color):
        dx = tip[0] - cp_before[0]
        dy = tip[1] - cp_before[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1.0:
            return
        nx, ny = dx / dist, dy / dist
        px, py = -ny, nx

        al = _style.arrow_length
        aw = _style.arrow_half_width
        base_center = (tip[0] - nx * al, tip[1] - ny * al)
        left = (base_center[0] + px * aw, base_center[1] + py * aw)
        right = (base_center[0] - px * aw, base_center[1] - py * aw)

        draw_list.add_triangle_filled(tip, left, right, color)

    # Node drawing
    def _draw_node(self, draw_list, name, pos, radius, base_color, activation=0.0):
        r = radius
        ox, oy = _style.node_shadow_offset
        act = float(np.clip(activation, 0.0, 1.0))
        act_visual = act ** 0.35
        theme = _theme_colors()

        warm = _style.col_activation
        br, bg, bb = base_color[:3]
        fill_r = br + (warm[0] - br) * act_visual * 0.55
        fill_g = bg + (warm[1] - bg) * act_visual * 0.55
        fill_b = bb + (warm[2] - bb) * act_visual * 0.55

        col_fill = _col32((fill_r, fill_g, fill_b, max(act, 0.5)))
        col_border = _col32(theme["border"])
        col_shadow = _col32((*theme["shadow"], _style.node_shadow_alpha))

        # Hover tooltip (shows full name)
        if imgui.is_mouse_hovering_rect(
            imgui.ImVec2(pos[0] - r, pos[1] - r),
            imgui.ImVec2(pos[0] + r, pos[1] + r),
        ):
            imgui.begin_tooltip()
            imgui.text(name)
            imgui.text(f"act: {act:.3f}")
            imgui.end_tooltip()

        # Glow
        if act > 0.1:
            for glow_r, glow_a in ((r + 7, 0.04), (r + 4, 0.08)):
                draw_list.add_circle_filled(
                    pos, glow_r,
                    _col32((*warm, glow_a * act_visual)),
                )

        # Drop shadow
        draw_list.add_circle_filled(
            (pos[0] + ox, pos[1] + oy), r + 1, col_shadow,
        )

        # Fill
        draw_list.add_circle_filled(pos, r, col_fill)

        # Inner rim
        draw_list.add_circle(pos, r - 1.0, _col32((0.0, 0.0, 0.0, 0.06)), 48, 1.5)

        # Border
        draw_list.add_circle(pos, r, col_border, 48, _style.node_border_thickness)

        # Small specular glint
        draw_list.add_circle_filled(
            (pos[0] - r * 0.25, pos[1] - r * 0.25), r * 0.2,
            _col32((*theme["highlight"], 0.3)),
        )
