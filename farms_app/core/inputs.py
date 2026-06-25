""" Input management: global shortcuts and input helpers """


from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from imgui_bundle import imgui

if TYPE_CHECKING:
    from farms_app.core.extension import Extension


class Shortcut:
    """A registered global shortcut."""

    def __init__(self, key: imgui.Key, callback: Callable, description: str = "",
                 mod_ctrl: bool = False, mod_shift: bool = False, mod_alt: bool = False):
        self.key = key
        self.callback = callback
        self.description = description
        self.mod_ctrl = mod_ctrl
        self.mod_shift = mod_shift
        self.mod_alt = mod_alt


class InputManager:
    """Global shortcut dispatch and input helpers.

    Processes global shortcuts before extensions see input.
    Extensions handle their own input in on_event() using ImGui state directly.

    Future: event layers (global -> focused extension -> focused window)
    with propagation/consumption semantics.
    """

    def __init__(self):
        self._shortcuts: list[Shortcut] = []

    def register(self, key: imgui.Key, callback: Callable, description: str = "",
                 mod_ctrl: bool = False, mod_shift: bool = False, mod_alt: bool = False):
        """Register a global shortcut."""
        self._shortcuts.append(Shortcut(
            key=key, callback=callback, description=description,
            mod_ctrl=mod_ctrl, mod_shift=mod_shift, mod_alt=mod_alt,
        ))

    def process(self):
        """Check and fire global shortcuts. Call once per frame before extension on_event()."""
        io = imgui.get_io()
        # Don't process shortcuts when typing in an input field or when a modal is open
        if io.want_text_input:
            return
        if imgui.is_popup_open("", imgui.PopupFlags_.any_popup_id):
            return

        for shortcut in self._shortcuts:
            if not imgui.is_key_pressed(shortcut.key):
                continue
            if shortcut.mod_ctrl and not io.key_ctrl:
                continue
            if shortcut.mod_shift and not io.key_shift:
                continue
            if shortcut.mod_alt and not io.key_alt:
                continue
            shortcut.callback()

    @staticmethod
    def is_extension_focused(extension: Extension) -> bool:
        """True if any of the extension's windows are focused."""
        for window in extension.windows.values():
            if not window._initialized:
                continue
            # Check if this window's imgui window is focused
            # ImGui tracks focus by window name
            if imgui.is_window_focused(imgui.FocusedFlags_.none):
                return True
        return False

    @staticmethod
    def is_extension_hovered(extension: Extension) -> bool:
        """True if any of the extension's windows are hovered."""
        for window in extension.windows.values():
            if not window._initialized:
                continue
            if imgui.is_window_hovered(imgui.HoveredFlags_.none):
                return True
        return False
