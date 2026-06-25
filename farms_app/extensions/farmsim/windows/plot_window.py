""" Configurable plot window for FARMS simulation data.

Renders plots based on a PlotWindowConfig — users can add, remove,
and reconfigure plots at runtime via an inline configuration panel.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from farms_app.core.window import Window
from imgui_bundle import imgui, implot


PLOT_TYPES = ["time_series", "xy"]
LAYOUT_TYPES = ["subplots_vertical", "tabs"]


# ── Configuration ────────────────────────────────────────────────────

@dataclass
class PlotConfig:
    """Configuration for a single plot."""
    plot_type: str = "time_series"   # "time_series" or "xy"
    x_source: str = ""               # data source name (ignored for time_series)
    y_sources: list[str] = field(default_factory=list)
    title: str = ""
    y_label: str = ""


@dataclass
class PlotWindowConfig:
    """Configuration for a plot window."""
    name: str = "Plot"
    layout: str = "subplots_vertical"   # "subplots_vertical", "tabs"
    plots: list[PlotConfig] = field(default_factory=list)


# ── Window ───────────────────────────────────────────────────────────

class PlotWindow(Window):
    """A window that renders plots based on its configuration.

    Data is fetched from the extension's data registry each frame.
    Toggle the config panel with the gear button to add/remove/reconfigure plots.
    """

    def __init__(self, extension, config: PlotWindowConfig):
        super().__init__(config.name, extension)
        self.config = config
        self._time_axis = np.linspace(-1.0, 0.0, 1000)
        self._show_config = False
        self._plot_to_remove = None

    def on_render(self):
        task = self._extension.task
        if task is None:
            imgui.text("No simulation loaded")
            return

        # Config toggle button
        if imgui.button("Config" if not self._show_config else "Close"):
            self._show_config = not self._show_config

        if self._show_config:
            self._render_config_panel()
            return

        # Plots
        if self.config.layout == "tabs":
            self._render_tabs(task)
        else:
            self._render_subplots(task)

    # ── Configuration panel ──────────────────────────────────────────

    def _render_config_panel(self):
        registry = self._extension.registry

        # Layout selector
        layout_idx = LAYOUT_TYPES.index(self.config.layout) if self.config.layout in LAYOUT_TYPES else 0
        changed, layout_idx = imgui.combo("Layout", layout_idx, LAYOUT_TYPES)
        if changed:
            self.config.layout = LAYOUT_TYPES[layout_idx]

        imgui.separator()

        # Per-plot config
        self._plot_to_remove = None
        for i, plot_cfg in enumerate(self.config.plots):
            imgui.push_id(i)
            self._render_plot_config(i, plot_cfg, registry)
            imgui.pop_id()
            imgui.separator()

        # Remove deferred
        if self._plot_to_remove is not None and self._plot_to_remove < len(self.config.plots):
            self.config.plots.pop(self._plot_to_remove)

        # Add plot button
        if imgui.button("+ Add Plot"):
            self.config.plots.append(PlotConfig(title=f"Plot {len(self.config.plots) + 1}"))

    def _render_plot_config(self, index, plot_cfg, registry):
        expanded = imgui.tree_node(f"{plot_cfg.title or f'Plot {index + 1}'}##plot_{index}")
        if not expanded:
            return

        # Title
        changed, new_title = imgui.input_text("Title", plot_cfg.title)
        if changed:
            plot_cfg.title = new_title

        # Plot type
        type_idx = PLOT_TYPES.index(plot_cfg.plot_type) if plot_cfg.plot_type in PLOT_TYPES else 0
        changed, type_idx = imgui.combo("Type", type_idx, PLOT_TYPES)
        if changed:
            plot_cfg.plot_type = PLOT_TYPES[type_idx]

        # X source (for xy plots)
        if plot_cfg.plot_type == "xy":
            self._render_source_selector("X axis", plot_cfg, "x_source", registry)

        # Y sources
        imgui.text("Signals:")
        sources_to_remove = []
        for j, source_name in enumerate(plot_cfg.y_sources):
            imgui.push_id(j)
            imgui.bullet_text(source_name)
            imgui.same_line()
            if imgui.small_button("x"):
                sources_to_remove.append(j)
            imgui.pop_id()

        for j in reversed(sources_to_remove):
            plot_cfg.y_sources.pop(j)

        # Add signal picker
        self._render_add_signal(plot_cfg, registry)

        # Remove plot
        imgui.spacing()
        if imgui.small_button("Remove Plot"):
            self._plot_to_remove = index

        imgui.tree_pop()

    def _render_source_selector(self, label, plot_cfg, attr, registry):
        """Button + popup to select a single source for an attribute."""
        current = getattr(plot_cfg, attr)
        popup_id = f"##src_popup_{label}_{id(plot_cfg)}"

        imgui.text(f"{label}:")
        imgui.same_line()
        if imgui.button(current if current else "(select)##" + label):
            imgui.open_popup(popup_id)

        if imgui.begin_popup(popup_id):
            self._render_source_tree(registry, plot_cfg, attr=attr)
            imgui.end_popup()

    def _render_add_signal(self, plot_cfg, registry):
        """Button + popup to add signals to y_sources."""
        popup_id = f"##add_sig_{id(plot_cfg)}"

        if imgui.button("+ Add Signal"):
            imgui.open_popup(popup_id)

        if imgui.begin_popup(popup_id):
            self._render_source_tree(registry, plot_cfg, attr=None)
            imgui.end_popup()

    def _render_source_tree(self, registry, plot_cfg, attr=None):
        """Grouped tree of data sources. If attr is set, clicking sets that
        attribute (single select). If attr is None, clicking appends to y_sources."""
        for group_name in registry.groups:
            if imgui.tree_node(group_name):
                items = {}
                for source in registry.group(group_name):
                    parts = source.name.split("/")
                    item_name = parts[1] if len(parts) > 2 else parts[-1]
                    if item_name not in items:
                        items[item_name] = []
                    items[item_name].append(source)

                for item_name, sources in items.items():
                    if imgui.tree_node(item_name):
                        for source in sources:
                            channel = source.name.rsplit("/", 1)[-1]
                            display = f"{channel} ({source.unit})" if source.unit else channel

                            if attr is None:
                                # Multi-select mode: add to y_sources
                                already = source.name in plot_cfg.y_sources
                                if already:
                                    imgui.text_disabled(display)
                                elif imgui.menu_item_simple(display):
                                    plot_cfg.y_sources.append(source.name)
                            else:
                                # Single-select mode: set attribute
                                current = getattr(plot_cfg, attr)
                                if imgui.menu_item_simple(display):
                                    setattr(plot_cfg, attr, source.name)
                                    imgui.close_current_popup()
                        imgui.tree_pop()
                imgui.tree_pop()

    # ── Plot rendering ───────────────────────────────────────────────

    def _render_subplots(self, task):
        plots = self.config.plots
        if not plots:
            imgui.text("No plots configured — click Config to add plots")
            return

        flags = (
            implot.SubplotFlags_.no_title |
            implot.SubplotFlags_.link_cols
        )
        if implot.begin_subplots(
            f"##{self.config.name}_subplots",
            rows=len(plots), cols=1,
            size=imgui.ImVec2(-1, -1),
            flags=flags,
        ):
            for plot_cfg in plots:
                self._render_plot(plot_cfg, task)
            implot.end_subplots()

    def _render_tabs(self, task):
        if imgui.begin_tab_bar(f"##{self.config.name}_tabs"):
            for plot_cfg in self.config.plots:
                if imgui.begin_tab_item(plot_cfg.title or "Plot")[0]:
                    self._render_plot(plot_cfg, task)
                    imgui.end_tab_item()
            imgui.end_tab_bar()

    def _render_plot(self, plot_cfg: PlotConfig, task):
        if plot_cfg.plot_type == "time_series":
            self._render_time_series(plot_cfg, task)
        elif plot_cfg.plot_type == "xy":
            self._render_xy(plot_cfg, task)

    # ── Plot type renderers ──────────────────────────────────────────

    def _render_time_series(self, plot_cfg: PlotConfig, task):
        registry = self._extension.registry
        iteration = task.iteration
        plot_flags = implot.Flags_.no_box_select | implot.Flags_.no_menus

        title = plot_cfg.title or "##ts"
        if implot.begin_plot(title, flags=plot_flags):
            axis_flags_x = (
                implot.AxisFlags_.no_tick_labels |
                implot.AxisFlags_.no_tick_marks
            )
            axis_flags_y = (
                implot.AxisFlags_.range_fit
                # implot.AxisFlags_.auto_fit |
            )
            implot.setup_axis(implot.ImAxis_.x1, "Time", flags=axis_flags_x)
            implot.setup_axis(implot.ImAxis_.y1, plot_cfg.y_label, flags=axis_flags_y)
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1, -1.0, 0.0)

            for source_name in plot_cfg.y_sources:
                source = registry.get(source_name)
                if source is None:
                    continue
                data = np.ascontiguousarray(
                    np.roll(source.accessor(), -iteration)
                )
                label = source_name.rsplit("/", 1)[-1]
                implot.plot_line(label, self._time_axis, data)

            implot.end_plot()

    def _render_xy(self, plot_cfg: PlotConfig, task):
        registry = self._extension.registry
        iteration = task.iteration

        x_source = registry.get(plot_cfg.x_source)
        if x_source is None:
            if implot.begin_plot(plot_cfg.title or "##xy"):
                implot.end_plot()
            return

        title = plot_cfg.title or "##xy"
        if implot.begin_plot(title, flags=implot.Flags_.no_box_select):
            x_label = plot_cfg.x_source.rsplit("/", 1)[-1]
            implot.setup_axis(implot.ImAxis_.x1, x_label)
            implot.setup_axis(implot.ImAxis_.y1, plot_cfg.y_label)

            x_data = np.ascontiguousarray(
                np.roll(x_source.accessor(), -iteration)
            )

            for source_name in plot_cfg.y_sources:
                source = registry.get(source_name)
                if source is None:
                    continue
                y_data = np.ascontiguousarray(
                    np.roll(source.accessor(), -iteration)
                )
                label = source_name.rsplit("/", 1)[-1]
                implot.plot_line(label, x_data, y_data)

            implot.end_plot()
