""" Logger """

import time
from argparse import ArgumentParser

import networkx as nx
import numpy as np
from farms_app.core.extension import CustomExtension
from farms_app.core.window import BaseWindow
from farms_app.utils import colors
from farms_core import pylog
from farms_core.io.yaml import read_yaml
from farms_network.core.network import Network
from farms_network.core.options import NetworkOptions
from imgui_bundle import imgui, imgui_ctx, implot, implot3d
from imgui_bundle import portable_file_dialogs as pfd
from tqdm import tqdm


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


def compute_phases(times, data):
    phases = (np.array(data) > 0.1).astype(np.int16)
    # phases = np.logical_not(phases).astype(np.int16)
    phases_xs = []
    phases_ys = []
    for j in range(len(data)):
        phases_start = np.where(np.diff(phases[j, :], prepend=0) == 1.0)[0]
        phases_ends = np.where(np.diff(phases[j, :], append=0) == -1.0)[0]
        phases_xs.append(np.vstack(
            (times[phases_start], times[phases_start], times[phases_ends], times[phases_ends])
        ).T)
        phases_ys.append(np.ones(np.shape(phases_xs[j]))*j)
        # if np.all(len(phases_start) > 3):
        phases_ys[j][:, 1] += 1
        phases_ys[j][:, 2] += 1

    return phases_xs, phases_ys


def add_plot(iteration, data):
    """ """
    # times = data.times.array[iteration%1000:]
    side = "right"
    limb = "fore"
    plot_names = [
        f"{side}_{limb}_RG_E",
        f"{side}_{limb}_RG_F",
        f"left_fore_RG_F",
        f"right_hind_RG_F",
        f"left_hind_RG_F",
        f"{side}_{limb}_PF_FA",
        f"{side}_{limb}_PF_EA",
        f"{side}_{limb}_PF_FB",
        f"{side}_{limb}_PF_EB",
        f"{side}_{limb}_RG_F_DR",
    ]

    plot_labels = [
        "RH_RG_E",
        "RH_RG_F",
        "LF_RG_F",
        "RH_RG_F",
        "LH_RG_F",
        "RH_PF_FA",
        "RH_PF_EA",
        "RH_PF_FB",
        "RH_PF_EB",
        "RH_RG_F_DR",
    ]

    nodes_names = [
        node.name
        for node in data.nodes
    ]

    plot_nodes = [
        nodes_names.index(name)
        for name in plot_names
        if name in nodes_names
    ]
    if not plot_nodes:
        return

    outputs = np.vstack(
        (
            *[
                data.nodes[plot_nodes[j]].output.values
                for j in range(len(plot_nodes))
            ],
            data.nodes[plot_nodes[-1]].external_input.values,
        )
    )
    if iteration < 1000:
        plot_data = np.array(outputs[:, :iteration])
    else:
        plot_data = np.array(outputs[:, iteration-1000:iteration])
    # plot_data = np.vstack((outputs[iteration%1000:], outputs[:iteration%1000]))

    times = np.array((np.linspace(0.0, 1.0, 1000)*-1.0)[::-1])

    phases_xs, phases_ys = compute_phases(times, plot_data[1:5, :])

    # phases = (np.array(plot_data[0, :]) > 0.1).astype(np.int16)
    # phases = np.logical_not(phases).astype(np.int16)
    # phases_start = np.where(np.diff(phases, prepend=0) == 1.0)[0]
    # phases_ends = np.where(np.diff(phases, append=0) == -1.0)[0]
    # phases_xs = np.vstack(
    #     (times[phases_start], times[phases_start], times[phases_ends], times[phases_ends])
    # ).T
    # phases_ys = np.ones(np.shape(phases_xs))
    # if len(phases_start) > 3:
    #     phases_ys[:, 1] += 1.0
    #     phases_ys[:, 2] += 1.0

    colors = {
        "RF": imgui.IM_COL32(28, 107, 180, 255),
        "LF": imgui.IM_COL32(23, 163, 74, 255),
        "RH": imgui.IM_COL32(200, 38, 39, 255),
        "LH":  imgui.IM_COL32(255, 252, 212, 255),  # imgui.IM_COL32(0, 0, 0, 255),
        "right_fore_RG_F": imgui.IM_COL32(28, 107, 180, 255),
        "left_fore_RG_F": imgui.IM_COL32(23, 163, 74, 255),
        "right_hind_RG_F": imgui.IM_COL32(200, 38, 39, 255),
        "left_hind_RG_F": imgui.IM_COL32(255, 252, 212, 255), #  imgui.IM_COL32(0, 0, 0, 255),
    }
    if implot.begin_subplots(
            "Network Activity",
            3,
            1,
            imgui.ImVec2(-1, -1),
            row_col_ratios=implot.SubplotsRowColRatios(row_ratios=[0.1, 0.8, 0.1], col_ratios=[1])
    ):
        if implot.begin_plot(""):
            flags = (
                implot.AxisFlags_.no_label | implot.AxisFlags_.no_tick_labels | implot.AxisFlags_.no_tick_marks
            )
            implot.setup_axis(implot.ImAxis_.y1, "Drive")
            implot.setup_axis(implot.ImAxis_.x1, flags=flags)
            implot.setup_axis_links(implot.ImAxis_.x1, implot.BoxedValue(-1.0), implot.BoxedValue(0.0))
            implot.setup_axis_limits(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits(implot.ImAxis_.y1, 0.0, 1.5)
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits_constraints(implot.ImAxis_.y1, 0.0, 1.5)
            implot.plot_line("RG-F-Dr", times, plot_data[-1, :])
            implot.end_plot()
        if implot.begin_plot(""):
            implot.setup_axis(implot.ImAxis_.y1, "Activity")
            implot.setup_axis(
                implot.ImAxis_.x1,
                flags=(
                    implot.AxisFlags_.no_tick_labels |
                    implot.AxisFlags_.no_tick_marks
                )
            )
            implot.setup_axis_links(implot.ImAxis_.x1, implot.BoxedValue(-1.0), implot.BoxedValue(0.0))
            implot.setup_axis_limits(implot.ImAxis_.y1, -1*len(plot_names), 1.0)
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits_constraints(implot.ImAxis_.y1, -8.2, 1.0)
            implot.setup_axis_ticks(
                axis=implot.ImAxis_.y1,
                v_min=-8.0,
                v_max=0.0,
                n_ticks=int(len(plot_names[:-1])),
                labels=(plot_labels[:-1])[::-1],
                keep_default=False
            )
            for j in range(len(plot_nodes[:-1])):
                if plot_names[j] in colors:
                    implot.push_style_color(implot.Col_.line, colors.get(plot_names[j]))
                    implot.plot_line(plot_names[j], times, plot_data[j, :] - j)
                    implot.pop_style_color()
                else:
                    implot.plot_line(plot_names[j], times, plot_data[j, :] - j)
            implot.end_plot()
        if len(plot_nodes) > 7:
            if implot.begin_plot("", flags=implot.Flags_.no_legend):
                implot.setup_axis_limits(implot.ImAxis_.x1, -1.0, 0.0)
                implot.setup_axis_limits(implot.ImAxis_.y1, 0.0, 4.0)
                implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
                implot.setup_axis_limits_constraints(implot.ImAxis_.y1, 0.0, 4.0)
                implot.setup_axis(implot.ImAxis_.y1, flags=implot.AxisFlags_.invert)
                implot.setup_axis_ticks(
                    axis=implot.ImAxis_.y1,
                    v_min=0.5,
                    v_max=3.5,
                    n_ticks=int(4),
                    labels=("RF", "LF", "RH", "LH"),
                    keep_default=False
                )
                for j, limb in enumerate(("RF", "LF", "RH", "LH")):
                    # if len(phases_xs[j]) > 3:
                    implot.push_style_color(
                        implot.Col_.fill,
                        colors[limb]
                    )
                    implot.plot_shaded(
                        limb,
                        phases_xs[j].flatten(),
                        phases_ys[j].flatten(),
                        yref=j
                    )
                    implot.pop_style_color()
                implot.end_plot()
        implot.end_subplots()


def draw_muscle_activity(iteration, data, plot_nodes, plot_names, title):

    outputs = np.vstack(
        [
            data.nodes[plot_nodes[j]].output.array
            for j in range(len(plot_nodes))
        ]
    )
    if iteration < 1000:
        plot_data = np.array(outputs[:, :iteration])
    else:
        plot_data = np.array(outputs[:, iteration-1000:iteration])

    times = np.array((np.linspace(0.0, 1.0, 1000)*-1.0)[::-1])

    with imgui_ctx.begin(title):
        if implot.begin_plot("Muscle Activity", imgui.ImVec2(-1, -1)):
            implot.setup_axis(implot.ImAxis_.x1, "Time")
            implot.setup_axis(implot.ImAxis_.y1, "Activity")
            implot.setup_axis_links(implot.ImAxis_.x1, implot.BoxedValue(-1.0), implot.BoxedValue(0.0))
            implot.setup_axis_limits(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits(implot.ImAxis_.y1, -1*(len(plot_nodes)-1), 1.0)
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_ticks(
                axis=implot.ImAxis_.y1,
                v_min=-1*(len(plot_nodes)-1),
                v_max=0.0,
                n_ticks=int(len(plot_names)),
                labels=plot_names[::-1],
                keep_default=False
            )
            # implot.setup_axis_limits_constraints(implot.ImAxis_.y1, -5.2, 1.0)
            for j in range(len(plot_nodes)):
                implot.plot_line(plot_names[j], times, plot_data[j, :] - j)
            implot.end_plot()


def plot_hind_motor_activity(iteration, data, side="right"):
    side = "left"
    limb = "hind"

    muscle_names = [
        "bfa",
        "ip",
        "bfpst",
        "rf",
        "va",
        "mg",
        "sol",
        "ta",
        "ab",
        "gm_dorsal",
        "edl",
        "fdl",
    ]

    nodes_names = [
        node.name
        for node in data.nodes
    ]

    plot_nodes = [
        nodes_names.index(f"{side}_{limb}_{name}_Mn")
        for name in muscle_names
        if f"{side}_{limb}_{name}_Mn" in nodes_names
    ]
    draw_muscle_activity(iteration, data, plot_nodes, muscle_names, title="Hindlimb muscles")


def plot_fore_motor_activity(iteration, data, side="right"):
    side = "right"
    limb = "fore"

    muscle_names = [
        "spd",
        "ssp",
        "abd",
        "add",
        "tbl",
        "tbo",
        "bbs",
        "bra",
        "eip",
        "fcu",
    ]

    nodes_names = [
        node.name
        for node in data.nodes
    ]

    plot_nodes = [
        nodes_names.index(f"{side}_{limb}_{name}_Mn")
        for name in muscle_names
        if f"{side}_{limb}_{name}_Mn" in nodes_names
    ]

    draw_muscle_activity(iteration, data, plot_nodes, muscle_names, title="Forelimb muscles")


def __draw_muscle_activity(iteration, data):
    """ Draw muscle activity """
    side = "left"
    limb = "hind"

    muscle_names = [
        "bfa",
        "ip",
        "bfpst",
        "rf",
        "va",
        "mg",
        "sol",
        "ta",
        "ab",
        "gm_dorsal",
        "edl",
        "fdl",
    ]

    nodes_names = [
        node.name
        for node in data.nodes
    ]

    plot_nodes = [
        nodes_names.index(f"{side}_{limb}_{name}_Mn")
        for name in muscle_names
    ]
    if not plot_nodes:
        return
    outputs = np.vstack(
        [
            data.nodes[plot_nodes[j]].output
            for j in range(len(plot_nodes))
        ]
    )
    if iteration < 1000:
        plot_data = np.array(outputs[:, :iteration])
    else:
        plot_data = np.array(outputs[:, iteration-1000:iteration])

    times = np.array((np.linspace(0.0, 1.0, 1000)*-1.0)[::-1])

    with imgui_ctx.begin("Muscle activity"):
        if implot.begin_plot("Muscle Activity", imgui.ImVec2(-1, -1)):
            implot.setup_axis(implot.ImAxis_.x1, "Time")
            implot.setup_axis(implot.ImAxis_.y1, "Activity")
            implot.setup_axis_links(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits(implot.ImAxis_.y1, -1*len(plot_nodes), 1.0)
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
            # implot.setup_axis_limits_constraints(implot.ImAxis_.y1, -5.2, 1.0)
            for j in range(len(plot_nodes)):
                implot.plot_line(muscle_names[j], times, plot_data[j, :] - j)
            implot.end_plot()

    # plot_nodes = [
    #     nodes_names.index(f"{side}_{limb}_{name}_Rn")
    #     for name in muscle_names
    # ]
    # if not plot_nodes:
    #     return
    # outputs = np.vstack(
    #     [
    #         data.nodes[plot_nodes[j]].output
    #         for j in range(len(plot_nodes))
    #     ]
    # )
    # if iteration < 1000:
    #     plot_data = np.array(outputs[:, :iteration])
    # else:
    #     plot_data = np.array(outputs[:, iteration-1000:iteration])

    # with imgui_ctx.begin("Renshaw activity"):
    #     if implot.begin_plot("Renshaw Activity", imgui.ImVec2(-1, -1)):
    #         implot.setup_axis(implot.ImAxis_.x1, "Time")
    #         implot.setup_axis(implot.ImAxis_.y1, "Activity")
    #         implot.setup_axis_links(implot.ImAxis_.x1, -1.0, 0.0)
    #         implot.setup_axis_limits(implot.ImAxis_.x1, -1.0, 0.0)
    #         implot.setup_axis_limits(implot.ImAxis_.y1, -1*len(plot_nodes), 1.0)
    #         implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
    #         # implot.setup_axis_limits_constraints(implot.ImAxis_.y1, -5.2, 1.0)
    #         for j in range(len(plot_nodes)):
    #             implot.plot_line(muscle_names[j], times, plot_data[j, :] - j)
    #         implot.end_plot()

    Ia_In_names = ("EA", "EB", "FA", "FB")
    plot_nodes = [
        nodes_names.index(f"{side}_{limb}_Ia_In_{name}")
        for name in ("EA", "EB", "FA", "FB")
    ]
    plot_nodes = [
        nodes_names.index(name)
        for name in nodes_names
        if "Ib_In_e" in name
    ]
    plot_labels = [
        name
        for name in nodes_names
        if "Ib_In_e" in name
    ]
    if not plot_nodes:
        return
    outputs = np.vstack(
        [
            data.nodes[plot_nodes[j]].output
            for j in range(len(plot_nodes))
        ]
    )
    if iteration < 1000:
        plot_data = np.array(outputs[:, :iteration])
    else:
        plot_data = np.array(outputs[:, iteration-1000:iteration])

    with imgui_ctx.begin("Sensory interneuron activity"):
        if implot.begin_plot("Sensory interneuron Activity", imgui.ImVec2(-1, -1)):
            implot.setup_axis(implot.ImAxis_.x1, "Time")
            implot.setup_axis(implot.ImAxis_.y1, "Activity")
            implot.setup_axis_links(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits(implot.ImAxis_.y1, -1*len(plot_nodes), 1.0)
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
            # implot.setup_axis_limits_constraints(implot.ImAxis_.y1, -5.2, 1.0)
            for j in range(len(plot_nodes)):
                implot.plot_line(plot_labels[j], times, plot_data[j, :] - j)
            implot.end_plot()


def draw_vn_activity(iteration, data):
    """ Draw muscle activity """
    side = "left"
    limb = "hind"

    vn_names = [
        f"{side}_{rate}_{axis}_{direction}_In_Vn"
        for rate in ("position", "velocity")
        for direction in ("clock", "cclock")
        for axis in ("pitch", "roll")
        for side in ("left", "right")
    ]

    nodes_names = [
        node.name
        for node in data.nodes
    ]

    plot_nodes = [
        nodes_names.index(name)
        for name in vn_names
    ]
    if not plot_nodes:
        return
    outputs = np.vstack(

        [
            data.nodes[plot_nodes[j]].output.array
            for j in range(len(plot_nodes))
        ]
    )
    if iteration < 1000:
        plot_data = np.array(outputs[:, :iteration])
    else:
        plot_data = np.array(outputs[:, iteration-1000:iteration])

    times = np.array((np.linspace(0.0, 1.0, 1000)*-1.0)[::-1])

    with imgui_ctx.begin("Vestibular"):
        if implot.begin_plot("Vestibular Activity", imgui.ImVec2(-1, -1)):
            implot.setup_axis(implot.ImAxis_.x1, "Time")
            implot.setup_axis(implot.ImAxis_.y1, "Activity")
            implot.setup_axis_links(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits(implot.ImAxis_.x1, -1.0, 0.0)
            implot.setup_axis_limits(implot.ImAxis_.y1, -1*len(plot_nodes), 1.0)
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)
            # implot.setup_axis_limits_constraints(implot.ImAxis_.y1, -5.2, 1.0)
            for j in range(len(plot_nodes)):
                implot.plot_line(vn_names[j], times, plot_data[j, :] - j)
            implot.end_plot()


def draw_network(network_options, data, iteration, edges_x, edges_y):
    """ Draw network """

    nodes = network_options.nodes
    edges = network_options.edges

    imgui.WindowFlags_
    with imgui_ctx.begin("Full-Network"):
        flags = (
            implot.AxisFlags_.no_label |
            implot.AxisFlags_.no_tick_labels |
            implot.AxisFlags_.no_tick_marks
        )
        if implot.begin_plot(
                "vis", imgui.ImVec2((-1, -1)), implot.Flags_.equal
        ):
            implot.setup_axis(implot.ImAxis_.x1, flags=flags)
            implot.setup_axis(implot.ImAxis_.y1, flags=flags)
            implot.plot_line(
                "",
                xs=edges_x,
                ys=edges_y,
                flags=implot.LineFlags_.segments
            )
            radius = 0.1
            circ_x = radius*np.cos(np.linspace(-np.pi, np.pi, 50))
            circ_y = radius*np.sin(np.linspace(-np.pi, np.pi, 50))
            for index, node in enumerate(nodes):
                implot.set_next_marker_style(
                    size=10.0 # *node.vi3sual.radius
                )
                implot.push_style_var(
                    implot.StyleVar_.fill_alpha,
                    0.05+data.nodes[index].output.array[iteration-1]*3.0
                )
                implot.plot_scatter(
                    "##",
                    xs=np.array((node.visual.position[0],)),
                    ys=np.array((node.visual.position[1],)),
                )
                # implot.plot_line(
                #     "##",
                #     node.visual.position[0]+circ_x,
                #     node.visual.position[1]+circ_y
                # )
                implot.pop_style_var()
                # implot.push_plot_clip_rect()
                # position = implot.plot_to_pixels(implot.Point(node.visual.position[:2]))
                # radius = implot.plot_to_pixels(0.001, 0.001)
                # color = imgui.IM_COL32(255, 0, 0, 255)
                # implot.get_plot_draw_list().add_circle(position, radius[0], color)
                # implot.pop_plot_clip_rect()

                # implot.push_plot_clip_rect()
                # color = imgui.IM_COL32(
                #     100, 185, 0,
                #     int(255*(data.nodes[index].output[iteration]))
                # )
                # implot.get_plot_draw_list().add_circle_filled(position, 7.5, color)
                # implot.pop_plot_clip_rect()
                implot.plot_text(
                    node.visual.label.replace("\\textsubscript", "")[0],
                    node.visual.position[0],
                    node.visual.position[1],
                )

            implot.end_plot()


def draw_slider(
        label: str,
        name: str,
        values: list,
        min_value: float = 0.0,
        max_value: float = 1.0
):
    with imgui_ctx.begin(name):
        clicked, values[0] = imgui.slider_float(
            label="alpha",
            v=values[0],
            v_min=min_value,
            v_max=max_value,
        )
        clicked, values[1] = imgui.slider_float(
            label="drive",
            v=values[1],
            v_min=min_value,
            v_max=max_value,
        )
        clicked, values[2] = imgui.slider_float(
            label="Ia",
            v=values[2],
            v_min=min_value,
            v_max=max_value,
        )
        clicked, values[3] = imgui.slider_float(
            label="II",
            v=values[3],
            v_min=min_value,
            v_max=max_value,
        )
        clicked, values[4] = imgui.slider_float(
            label="Ib",
            v=values[4],
            v_min=min_value,
            v_max=max_value,
        )
        clicked, values[5] = imgui.slider_float(
            label="Vn",
            v=values[5],
            v_min=-1.0,
            v_max=max_value,
        )
        clicked, values[6] = imgui.slider_float(
            label="Cut",
            v=values[6],
            v_min=min_value,
            v_max=max_value,
        )
    return values


def draw_table(network_options, network_data):
    """ Draw table """
    flags = (
        imgui.TableFlags_.borders | imgui.TableFlags_.row_bg | imgui.TableFlags_.resizable |
        imgui.TableFlags_.sortable
    )
    with imgui_ctx.begin("Table"):
        edges = network_options.edges
        nodes = network_options.nodes
        n_edges = len(edges)
        if imgui.begin_table("Edges", 3, flags):
            weights = network_data.connectivity.weights
            for col in ("Source", "Target", "Weight"):
                imgui.table_setup_column(col)
            imgui.table_headers_row()
            for row in range(n_edges):
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                imgui.text(edges[row].source)
                imgui.table_set_column_index(1)
                imgui.text(edges[row].target)
                imgui.table_set_column_index(2)
                imgui.push_id(row)
                # imgui.input_float("##row", weights[row])
                _, weights[row] = imgui.slider_float("##row", weights[row], -10.0, 10.0)
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


class DiagramStyle:
    node_radius = 12
    node_shadow_offset = (3, 3)
    node_shadow_alpha = 0.18
    node_highlight_alpha = 0.25
    curve_thickness = 3.0
    arrow_size = 8.0
    node_border_thickness = 2.3

    # colors
    col_node_blue = (0.68, 0.80, 1.0, 1.0)
    col_node_red = (1.0, 0.78, 0.78, 1.0)
    col_border_dark = (0.22, 0.22, 0.32, 1.0)
    col_edge = (0.18, 0.18, 0.18, 1.0)
    col_shadow = (0.0, 0.0, 0.0)
    col_highlight = (1.0, 1.0, 1.0)

style = DiagramStyle()


def col32(rgba):
    return imgui.color_convert_float4_to_u32(rgba)


class NetworkExtension(CustomExtension):
    """ Network """

    def __init__(self):
        name = "Network"
        super().__init__(name=name)

        self.network = None
        self.time = 0.0

        self.register_window(NetworkVisualizerWindow(self))
        self.register_window(NetworkPlotWindow(self))

        # # Integrate
        # self.N_ITERATIONS = self.network_options.integration.n_iterations
        # self.TIMESTEP = self.network_options.integration.timestep
        # self.BUFFER_SIZE = self.network_options.logs.buffer_size

        # self.inputs_view = self.network.data.external_inputs.array
        # self.drive_input = 0.0

        # edges_xy = np.array(
        #     [
        #         self.network_options.nodes[node_idx].visual.position[:2]
        #         for edge in self.network_options.edges
        #         for node_idx in (
        #                 self.network_options.nodes.index(edge.source),
        #                 self.network_options.nodes.index(edge.target),
        #         )
        #     ]
        # )
        # # for index in range(len(edges_xy) - 1):
        # #     edges_xy[index], edges_xy[index + 1] = connect_positions(
        # #         edges_xy[index+1], edges_xy[index], 0.1, 0.0
        # #     )
        # self.edges_x = np.array(edges_xy[:, 0])
        # self.edges_y = np.array(edges_xy[:, 1])

        # fps = 30.0
        # _time_draw = time.time()
        # _time_draw_last = _time_draw
        # _realtime = 0.1

        # io = imgui.get_io()

        # self.alpha_input_indices = [
        #     index
        #     for index, node in enumerate(self.network_options.nodes)
        #     if "input" in node.name and node.model == "external_relay"
        # ]
        # self.drive_input_indices = [
        #     index
        #     for index, node in enumerate(self.network_options.nodes)
        #     if "DR" in node.name and node.model == "linear"
        # ]
        # self.Ia_input_indices = [
        #     index
        #     for index, node in enumerate(self.network_options.nodes)
        #     if "Ia" == node.name[-2:]
        # ]
        # self.II_input_indices = [
        #     index
        #     for index, node in enumerate(self.network_options.nodes)
        #     if "II" == node.name[-2:]
        # ]
        # self.Ib_input_indices = [
        #     index
        #     for index, node in enumerate(self.network_options.nodes)
        #     if "Ib" == node.name[-2:]
        # ]
        # self.Vn_input_indices = [
        #     index
        #     for index, node in enumerate(self.network_options.nodes)
        #     if "Vn" == node.name[-2:] and node.model == "external_relay"
        # ]
        # self.Cut_input_indices = [
        #     index
        #     for index, node in enumerate(self.network_options.nodes)
        #     if "cut" == node.name[-3:] and node.model == "external_relay"
        # ]
        # self.slider_values = np.zeros((7,))
        # self.slider_values[0] = 1.0
        # self.input_array = np.zeros(np.shape(self.inputs_view))
        # self.input_array[self.alpha_input_indices] = 1.0
        # self.button_state = False2
        # # input_array[drive_input_indices[0]] *= 1.05
        # for index, node in enumerate(self.network_options.nodes):
        #     if "BS_DR" in node.name and node.model == "linear":
        #         bs_dr = index
        # self.iteration = 0
        # self.buffer_iteration = 0

    def on_update(self):
        if self.network:
            if (self.time < self.network.options.integration.n_iterations):
                self.network.step(self.time)
                self.network.update_logs(self.time)
                self.time += 1

    def render_menu(self):
        """ Render menu """
        imgui.begin_menu_bar()
        if imgui.begin_menu("File"):
            if imgui.menu_item_simple("Config"):
                result = pfd.open_file("Network config", filters=["*.yaml",]).result()
                if result:
                    try:
                        network_options = NetworkOptions.from_options(read_yaml(result[0]))
                        self.network = None
                    except KeyError:
                        pylog.error(f"Invalid network config {result[0]}")
                        raise KeyError

                    self.network = Network.from_options(network_options)
                    if self.network:
                        self.network.setup_integrator()
                        self.time = 0.0
                        self.windows[1].initialize()
                        self.windows[2].initialize()
            imgui.end_menu()
        imgui.end_menu_bar()

    def get_name(self) -> str:
        return "Network"

    def cleanup(self):
        """ Cleanup  """
        implot.plot_line
        pass

    def get_dependencies(self):
        return []

    # def render(self) -> None:
    #     if not self.show_window:
    #         return

    #     expanded, self.show_window = imgui.begin("Network", self.show_window)

    #     self.input_array[self.alpha_input_indices] = self.slider_values[0]
    #     self.input_array[self.drive_input_indices] = self.slider_values[1]
    #     self.input_array[self.Ia_input_indices] = self.slider_values[2]
    #     self.input_array[self.II_input_indices] = self.slider_values[3]
    #     self.input_array[self.Ib_input_indices] = self.slider_values[4]
    #     self.input_array[self.Vn_input_indices] = self.slider_values[5]
    #     self.input_array[self.Cut_input_indices] = self.slider_values[6]

    #     self.inputs_view[:] = self.input_array
    #     self.network.step()
    #     self.iteration += 1
    #     self.buffer_iteration = self.iteration%self.BUFFER_SIZE
    #     self.network.data.times.array[self.buffer_iteration] = (self.iteration)*self.TIMESTEP
    #     implot.push_style_var(implot.StyleVar_.line_weight, 2.0)
    #     self.slider_values = draw_slider(label="d", name="Drive", values=self.slider_values)
    #     add_plot(self.buffer_iteration, self.network.data)
    #     # button_state = draw_play_pause_button(button_state)
    #     draw_table(self.network_options, self.network.data)
    #     draw_network(self.network_options, self.network.data, self.buffer_iteration, self.edges_x, self.edges_y)
    #     plot_hind_motor_activity(self.buffer_iteration, self.network.data)
    #     plot_fore_motor_activity(self.buffer_iteration, self.network.data)
    #     # draw_vn_activity(self.buffer_iteration, network.data)
    #     implot.pop_style_var()

    #     imgui.end()


class NetworkPlotWindow(BaseWindow):

    def __init__(self, extension, network: Network = None):
        name: str = "plot"
        super().__init__(name, extension)

    def on_initialize(self):
        """ On initialize """
        self.iteration = 0
        self.network = self._extension.network

    def on_update(self):
        """ On update of the application """

    def on_render(self):
        """ Render main extension dockspace """
        add_plot(self.iteration, self.network.log)
        self.iteration += 1


class NetworkVisualizerWindow(BaseWindow):

    def __init__(self, extension, network: Network = None):
        name: str = "visualizer"
        super().__init__(name, extension)

    def on_initialize(self):
        """ On initialize """
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

    def draw_bezier_connection(self, draw_list, color):
      # soft cubic curve
      for p1, p2 in self.edges_p1p2:
        p1 = implot.plot_to_pixels(p1)
        p2 = implot.plot_to_pixels(p2)

        dx = (p2[0] - p1[0]) * 0.5
        cp1 = (p1[0] + dx, p1[1])
        cp2 = (p2[0] - dx, p2[1])
        draw_list.add_bezier_cubic(
            p1, cp1, cp2, p2,
            color, thickness=2.5
        )

    def on_update(self):
        """ On update of the application """

    def on_render(self):
        """ Render main extension dockspace """
        # _, self.network.data.nodes['BS_input'].external_input.values = imgui.slider_float(
        #     "Drive",
        #     self.network.data.nodes['BS_input'].external_input.values,
        #     0.0,
        #     1.5,
        # )
        _, self.network.data.nodes['BS_input_right'].external_input.values = imgui.slider_float(
            "Drive Right",
            self.network.data.nodes['BS_input_right'].external_input.values,
            0.0,
            1.5,
        )
        imgui.text(f"Time = {self._extension.time}")
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
          # implot.plot_line(
          #     "",
          #     xs=self.edges_x,
          #     ys=self.edges_y,
          #     flags=implot.LineFlags_.segments
          # )
          # radius = 0.1
          # circ_x = radius*np.cos(np.linspace(-np.pi, np.pi, 50))
          # circ_y = radius*np.sin(np.linspace(-np.pi, np.pi, 50))

          draw_list = implot.get_plot_draw_list()
          implot.push_plot_clip_rect()
          # safe to draw, nothing leaks outside plot
          self.draw_bezier_connection(draw_list, colors.COLORS['edge_strong'])
          implot.pop_plot_clip_rect()

          for index, node in enumerate(nodes):
              p1 = implot.plot_to_pixels(node.visual['position'][:2])

              implot.push_plot_clip_rect()
              # safe to draw, nothing leaks outside plot
              self.draw_node2(
                  node.name, draw_list, p1, 20, colors.rgb_u32(
                      *colors._RAW_COLORS['pastel_red'],
                      a=self.network.data.outputs.array[index]*3.0
                  )
              )
              draw_list.add_text(p1, colors.COLORS['edge'], node.visual['label'].replace("\\textsubscript", "")[0])
              implot.pop_plot_clip_rect()
            # implot.set_next_marker_style(size=15.0) # *node.vi3sual.radius
            # implot.push_style_var(implot.StyleVar_.fill_alpha, self.network.data.outputs.array[index])
            # implot.plot_scatter(
            #     "##",
            #     xs=np.array((node.visual['position'][0],)),
            #     ys=np.array((node.visual['position'][1],)),
            # )
            # implot.pop_style_var()
            # implot.set_next_marker_style(size=8.0) # *node.vi3sual.radius
            # implot.plot_scatter(
            #     "##",
            #     xs=np.array((node.visual['position'][0],)),
            #     ys=np.array((node.visual['position'][1],)),
            # )
            # implot.plot_text(
            #     node.visual['label'].replace("\\textsubscript", "")[0],
            #     node.visual['position'][0],
            #     node.visual['position'][1],
            # )

          implot.end_plot()


class NetworkVisualExtension(CustomExtension):
    """ Network """

    def __init__(self):
        self.show_window = True
        self.window_name = "Network"

        # run network
        self.network_options = NetworkOptions.from_options(
            read_yaml("/Users/tatarama/projects/work/research/neuromechanics/quadruped/mice/mouse-locomotion/data/config/muscles/quadruped_siggraph.yaml")
        )

        edges_xy = np.array(
            [
                self.network_options.nodes[node_idx].visual.position[:2]
                for edge in self.network_options.edges
                for node_idx in (
                    self.network_options.nodes.index(edge.source),
                    self.network_options.nodes.index(edge.target),
                )
            ]
        )
        self.edges_x = np.array(edges_xy[:, 0])
        self.edges_y = np.array(edges_xy[:, 1])

        self.graph = nx.node_link_graph(
            self.network.options,
            directed=True,
            multigraph=False,
            link="edges",
            name="name",
            source="source",
            target="target"
        )
        self.sparse_array = nx.to_scipy_sparse_array(self.graph)
        self.static = None

    def draw_network(self):
        """ Draw network """

        nodes = self.network_options.nodes
        edges = self.network_options.edges

        imgui.WindowFlags_
        flags = (
            implot.AxisFlags_.no_label |
            implot.AxisFlags_.no_tick_labels |
            implot.AxisFlags_.no_tick_marks
        )
        if imgui.begin_child("Network", size=(-1, -1), child_flags=imgui.ChildFlags_.resize_x | imgui.ChildFlags_.resize_y | imgui.ChildFlags_.borders):
            if implot.begin_plot("vis", size=(-1, -1), flags=implot.Flags_.equal):
                implot.setup_axis(implot.ImAxis_.x1, flags=flags)
                implot.setup_axis(implot.ImAxis_.y1, flags=flags)
                implot.plot_line(
                    "",
                    xs=self.edges_x,
                    ys=self.edges_y,
                    flags=implot.LineFlags_.segments
                )
                radius = 0.1
                circ_x = radius*np.cos(np.linspace(-np.pi, np.pi, 50))
                circ_y = radius*np.sin(np.linspace(-np.pi, np.pi, 50))
                for index, node in enumerate(nodes):
                    implot.set_next_marker_style(
                        size=10.0 # *node.vi3sual.radius
                    )
                    implot.push_style_var(implot.StyleVar_.fill_alpha, 1.0)
                    implot.plot_scatter(
                        "##",
                        xs=np.array((node.visual.position[0],)),
                        ys=np.array((node.visual.position[1],)),
                    )
                    # implot.plot_line(
                    #     "##",
                    #     node.visual.position[0]+circ_x,
                    #     node.visual.position[1]+circ_y
                    # )
                    implot.pop_style_var()
                    # implot.push_plot_clip_rect()
                    # position = implot.plot_to_pixels(implot.Point(node.visual.position[:2]))
                    # radius = implot.plot_to_pixels(0.001, 0.001)
                    # color = imgui.IM_COL32(255, 0, 0, 255)
                    # implot.get_plot_draw_list().add_circle(position, radius[0], color)
                    # implot.pop_plot_clip_rect()

                    # implot.push_plot_clip_rect()
                    # color = imgui.IM_COL32(
                    #     100, 185, 0,
                    #     int(255*(data.nodes[index].output[iteration]))
                    # )
                    # implot.get_plot_draw_list().add_circle_filled(position, 7.5, color)
                    # implot.pop_plot_clip_rect()
                    implot.plot_text(
                        node.visual.label.replace("\\textsubscript", "")[0],
                        node.visual.position[0],
                        node.visual.position[1],
                    )

                implot.end_plot()
            imgui.end_child()

    def draw_connectivity_map(self):
        """ Draw connectivity """
        axes_flags = implot.AxisFlags_.lock | implot.AxisFlags_.no_grid_lines | implot.AxisFlags_.no_tick_marks
        implot.push_colormap(implot.Colormap_.viridis)
        if imgui.begin_child("Network", size=(-1, -1), child_flags=imgui.ChildFlags_.resize_x | imgui.ChildFlags_.resize_y):
            if implot.begin_plot("Connectivity", flags=implot.Flags_.no_legend | implot.Flags_.no_mouse_text):
                # implot.setup_axes("", "", axes_flags, axes_flags)
                implot.setup_axis_ticks(implot.ImAxis_.x1, values=[j for j in range(10)], labels=[f"{j}" for j in range(10)], keep_default=False)
                implot.setup_axis_ticks(implot.ImAxis_.y1, values=[j for j in range(10)], labels=[f"{j}" for j in range(10)], keep_default=False)
                implot.plot_heatmap(
                    "network", self.sparse_array.todense(), label_fmt="%i"
                )
                implot.end_plot()
            imgui.end_child()
        implot.pop_colormap()


    def get_name(self) -> str:
        return "Network"

    def render_window(self) -> None:
        self.draw_network()
        # self.draw_connectivity_map()
