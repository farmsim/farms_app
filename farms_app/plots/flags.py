from functools import cache

from imgui_bundle import imgui, implot, em_to_vec2


@cache
def seaborn_style():
    """ Seaborn style """

    style = implot.get_style()
    style.set_color_(implot.Col_.line, implot.AUTO_COL)
    style.set_color_(implot.Col_.fill, implot.AUTO_COL)
    style.set_color_(implot.Col_.marker_outline, implot.AUTO_COL)
    style.set_color_(implot.Col_.marker_fill, implot.AUTO_COL)

    style.set_color_(implot.Col_.error_bar, imgui.ImVec4(0.00, 0.00, 0.00, 1.00))
    style.set_color_(implot.Col_.frame_bg, imgui.ImVec4(1.00, 1.00, 1.00, 1.00))
    style.set_color_(implot.Col_.plot_bg, imgui.ImVec4(0.92, 0.92, 0.95, 1.00))
    style.set_color_(implot.Col_.plot_border, imgui.ImVec4(0.00, 0.00, 0.00, 0.00))
    style.set_color_(implot.Col_.legend_bg, imgui.ImVec4(0.92, 0.92, 0.95, 1.00))
    style.set_color_(implot.Col_.legend_border, imgui.ImVec4(0.80, 0.81, 0.85, 1.00))
    style.set_color_(implot.Col_.legend_text, imgui.ImVec4(0.00, 0.00, 0.00, 1.00))
    style.set_color_(implot.Col_.title_text, imgui.ImVec4(0.00, 0.00, 0.00, 1.00))
    style.set_color_(implot.Col_.inlay_text, imgui.ImVec4(0.00, 0.00, 0.00, 1.00))
    style.set_color_(implot.Col_.axis_text, imgui.ImVec4(0.00, 0.00, 0.00, 1.00))
    style.set_color_(implot.Col_.axis_grid, imgui.ImVec4(1.00, 1.00, 1.00, 1.00))
    style.set_color_(implot.Col_.axis_bg_hovered, imgui.ImVec4(0.92, 0.92, 0.95, 1.00))
    style.set_color_(implot.Col_.axis_bg_active, imgui.ImVec4(0.92, 0.92, 0.95, 0.75))
    style.set_color_(implot.Col_.selection, imgui.ImVec4(1.00, 0.65, 0.00, 1.00))
    style.set_color_(implot.Col_.crosshairs, imgui.ImVec4(0.23, 0.10, 0.64, 0.50))

    style.line_weight = 1.5
    style.marker = implot.Marker_.none
    style.marker_size = 4
    style.marker_weight = 1
    style.fill_alpha = 1.0
    style.error_bar_size = 5
    style.error_bar_weight = 1.5
    style.digital_bit_height = 8
    style.digital_bit_gap = 4
    style.plot_border_size = 0
    style.minor_alpha = 1.0
    style.major_tick_len = imgui.ImVec2(0, 0)
    style.minor_tick_len = imgui.ImVec2(0, 0)
    style.major_tick_size = imgui.ImVec2(0, 0)
    style.minor_tick_size = imgui.ImVec2(0, 0)
    style.major_grid_size = imgui.ImVec2(1.2, 1.2)
    style.minor_grid_size = imgui.ImVec2(1.2, 1.2)
    style.plot_padding = em_to_vec2(0.75, 0.75)
    style.label_padding = em_to_vec2(0.3, 0.3)
    style.legend_padding = em_to_vec2(0.3, 0.3)
    style.mouse_pos_padding = em_to_vec2(0.3, 0.3)
    style.plot_min_size = em_to_vec2(19, 14)
