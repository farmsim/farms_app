""" Layout utilities for docking windows.

Thin wrapper around imgui.internal.dock_builder_* to keep
extension code clean and insulated from internal API details.
"""

import os

from imgui_bundle import imgui


def is_first_use() -> bool:
    """True when no imgui.ini exists (first launch)."""
    ini = imgui.get_io().get_ini_filename()
    if not ini:
        return True
    return not os.path.exists(ini)


def split(node_id: int, direction: imgui.Dir, ratio: float) -> tuple:
    """Split a dock node. Returns (id_at_direction, id_opposite)."""
    _, id_at_dir, id_opposite = imgui.internal.dock_builder_split_node_py(
        node_id, direction, ratio
    )
    return id_at_dir, id_opposite


def dock_window(window_id: str, node_id: int):
    """Assign a window to a dock node."""
    imgui.internal.dock_builder_dock_window(window_id, node_id)


def finish(node_id: int):
    """Finalize the dock builder layout."""
    imgui.internal.dock_builder_finish(node_id)
