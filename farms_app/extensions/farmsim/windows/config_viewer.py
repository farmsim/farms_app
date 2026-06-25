""" Read-only YAML config viewer for FARMS experiment options """

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from farms_app.core.window import Window
from farms_core.options import Options
from imgui_bundle import imgui

if TYPE_CHECKING:
    from farms_app.extensions.farmsim.extension import FARMSIMExtension


class ConfigViewerWindow(Window["FARMSIMExtension"]):
    """Displays the loaded experiment configuration as a navigable tree."""

    def __init__(self, extension):
        super().__init__("Config", extension)
        self._config_dict = None

    def set_config(self, exp_options):
        """Convert experiment options to a plain dict for display."""
        try:
            self._config_dict = {
                "simulation": _to_display(exp_options.simulation),
                "animats": [_to_display(a) for a in exp_options.animats],
                "arenas": [_to_display(a) for a in exp_options.arenas],
            }
        except Exception:
            self._config_dict = None

    def on_render(self):
        if self._config_dict is None:
            imgui.text("No experiment loaded")
            return

        _render_value("experiment", self._config_dict)


def _to_display(obj):
    """Convert an Options object (or any nested structure) to plain dicts/lists."""
    if isinstance(obj, Options):
        return {k: _to_display(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: _to_display(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_display(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _render_value(key, value, depth=0):
    """Recursively render a key-value pair as an ImGui tree."""
    if isinstance(value, dict):
        flags = imgui.TreeNodeFlags_.default_open if depth < 1 else 0
        if imgui.tree_node_ex(str(key), flags):
            for k, v in value.items():
                _render_value(k, v, depth + 1)
            imgui.tree_pop()

    elif isinstance(value, list):
        if len(value) == 0:
            imgui.text(f"{key}: []")
        elif isinstance(value[0], (dict, list)):
            # List of complex objects — show as indexed tree nodes
            if imgui.tree_node(str(key)):
                for i, item in enumerate(value):
                    _render_value(f"[{i}]", item, depth + 1)
                imgui.tree_pop()
        else:
            # List of primitives — show inline
            _render_leaf(key, value)

    else:
        _render_leaf(key, value)


def _render_leaf(key, value):
    """Render a single key: value line."""
    imgui.text_disabled(str(key))
    imgui.same_line()
    imgui.text(f": {_format_value(value)}")


def _format_value(value):
    """Format a value for display."""
    if isinstance(value, float):
        if abs(value) < 1e-3 and value != 0.0:
            return f"{value:.4e}"
        return f"{value:.6g}"
    if isinstance(value, list):
        if all(isinstance(v, (int, float)) for v in value):
            formatted = ", ".join(
                f"{v:.4g}" if isinstance(v, float) else str(v)
                for v in value
            )
            return f"[{formatted}]"
        return str(value)
    return str(value)
