""" Window """


from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Generic, Optional, TypeVar

from farms_app.console import console
from farms_core import pylog
from imgui_bundle import imgui, em_to_vec2

if TYPE_CHECKING:
    from farms_app.core.extension import Extension
    from farms_app.core.layout import Rect

E = TypeVar('E', bound='Extension')


class WindowManager:
    """Slim window registry. Will grow into layout persistence."""

    def __init__(self):
        self.windows: list[Window] = []

    def register(self, window: "Window") -> None:
        self.windows.append(window)

    def unregister(self, window: "Window") -> None:
        if window in self.windows:
            self.windows.remove(window)


class Window(Generic[E]):
    """Base class for all extension windows.

    Lifecycle
    ---------
    1. Construction  — sets up IDs and flags, no imgui calls.
    2. initialize()  — called explicitly by the extension after it has set up
                   its own context (e.g. state, runner).  Safe to call any
                   imgui API here since the manager guarantees a valid frame.
    3. _render()     — called every frame by Extension.render().
                   Guards on _initialized before touching imgui.

    Override on_initialize() for one-time setup.
    Override on_render() to draw window contents.
    """

    def __init__(
            self, name: str,
            extension: E,
            window_flags: imgui.WindowFlags_ = imgui.WindowFlags_.none,
            visible: bool = True,
    ):
        self.name: str = name
        self._extension: E = extension
        self._window_id: str = f"{name}##{self._extension.name}"
        self._window_flags: imgui.WindowFlags_ = window_flags

        # Window state
        self._initialized: bool = False
        self._visible: bool = visible
        self._focused: bool = False
        self._hovered: bool = False

        # Window properties that can be saved/restored
        self._window_size: Optional[tuple[float, float]] = None
        self._window_pos: Optional[tuple[float, float]] = None

        # Layout: optional default rect from RectCut (for floating windows)
        self._default_rect: Optional[Rect] = None
        # Layout reset: 0=idle, 2=undock next frame, 1=redock next frame
        self._reset_dock_phase: int = 0

    def initialize(self):
        """Initialize the window - called once after creation."""
        if self._initialized:
            return

        try:
            self.on_initialize()
            self._initialized = True
        except Exception as e:
            console.print_exception()
            pylog.error(f"Error initializing window {self._window_id}: {e}")

    def on_initialize(self):
        """Override for one-time setup. Called when extension says data is ready."""

    def on_render(self):
        """Override to draw window contents. Called inside imgui.begin()/end()."""

    def _render(self) -> None:
        """Internal render call — wraps on_render() in imgui.begin()/end()."""

        if not self._visible:
            self._reset_dock_phase = 0  # cancel pending reset
            return

        # Layout: _default_rect takes precedence over auto-docking.
        # Windows with a rect float at their RectCut position;
        # windows without one dock into the extension's dockspace.
        dock_id = self._extension.dockspace_id
        if self._reset_dock_phase == 2:
            imgui.set_next_window_dock_id(0, imgui.Cond_.always)
            self._reset_dock_phase = 1
        elif self._reset_dock_phase == 1:
            if dock_id:
                imgui.set_next_window_dock_id(dock_id, imgui.Cond_.always)
            self._reset_dock_phase = 0
        elif self._default_rect:
            r = self._default_rect
            imgui.set_next_window_pos(imgui.ImVec2(r.minx, r.miny), imgui.Cond_.first_use_ever)
            imgui.set_next_window_size(imgui.ImVec2(r.width, r.height), imgui.Cond_.first_use_ever)
        elif dock_id and self._extension.auto_dock_windows:
            imgui.set_next_window_dock_id(dock_id, imgui.Cond_.first_use_ever)
        else:
            imgui.set_next_window_size(em_to_vec2(25, 19), imgui.Cond_.first_use_ever)
        expanded, self._visible = imgui.begin(
            self._window_id,
            self._visible,
            flags=self._window_flags
        )
        self._focused = imgui.is_window_focused()
        self._hovered = imgui.is_window_hovered()
        if expanded:
            self.on_render()
        imgui.end()

    # Properties
    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visible = value

    @property
    def window_id(self) -> str:
        return self._window_id

    def show(self) -> None:
        self._visible = True

    def hide(self) -> None:
        self._visible = False

    def toggle_visibility(self) -> None:
        self._visible = not self._visible

    def set_window_flags(self, window_flags: imgui.WindowFlags_) -> None:
        self._window_flags = window_flags

    def set_window_size(self, width: float, height: float) -> None:
        self._window_size = (width, height)

    def set_window_pos(self, x: float, y: float) -> None:
        self._window_pos = (x, y)

    def get_state(self) -> Dict[str, Any]:
        """Get window state for persistence."""
        return {
            'visible': self._visible,
            'size': self._window_size,
            'pos': self._window_pos,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore window state from persistence."""
        self._visible = state.get('visible', True)
        self._window_size = state.get('size')
        self._window_pos = state.get('pos')
