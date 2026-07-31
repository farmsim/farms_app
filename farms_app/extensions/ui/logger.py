"""Logger — GUI log window using farms_core's GuiLogHandler."""

import logging

from colorama import Fore
from farms_app.core.extension import Extension
from farms_app.core.window import Window
from farms_core import pylog
from farms_core.pylog.log import GuiLogHandler, LogFormatter
from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd

# Map colorama Fore codes to RGBA for imgui rendering
_FORE_TO_RGBA = {
    Fore.CYAN:    (0.20, 0.80, 0.80, 1.0),
    Fore.GREEN:   (0.20, 0.80, 0.20, 1.0),
    Fore.YELLOW:  (0.80, 0.80, 0.20, 1.0),
    Fore.RED:     (0.80, 0.20, 0.20, 1.0),
    Fore.MAGENTA: (0.80, 0.30, 0.80, 1.0),
}

_DEFAULT_COLOR = (0.85, 0.85, 0.85, 1.0)


def _color_to_rgba(color):
    """Convert a colorama Fore string or RGBA tuple to an RGBA tuple."""
    if isinstance(color, tuple):
        return color
    return _FORE_TO_RGBA.get(color, _DEFAULT_COLOR)


_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
_LEVEL_VALUES = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]

_FILTER_LABELS = ['All', 'Debug', 'Info', 'Warning', 'Error']
_FILTER_VALUES = [logging.DEBUG, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]


class DebugExtension(Extension):
    """Logger extension — pipes pylog output into an imgui window."""

    def __init__(self):
        super().__init__(name="Debug")
        self.handler = GuiLogHandler()
        self.register_window(LoggerWindow(self))

    def on_enable(self):
        pylog.LOGGER.addHandler(self.handler)
        self.init_windows()

    def cleanup(self):
        pylog.LOGGER.removeHandler(self.handler)


class LoggerWindow(Window["DebugExtension"]):
    """Scrollable log window with level filtering and output controls."""

    def __init__(self, extension: DebugExtension):
        super().__init__(name="Log", extension=extension)
        self._auto_scroll = True
        self._filter_idx = 0
        self._filter_level = logging.DEBUG

        # Logger level control
        self._level_idx = _LEVELS.index(pylog.get_level().upper()) if pylog.get_level().upper() in _LEVELS else 0

        # Output toggles
        self._term_enabled = pylog.LOGGER.ch is not None
        self._file_enabled = pylog.LOGGER.fh is not None
        self._log_file_path = ""

    def on_render(self):
        handler: GuiLogHandler = self._extension.handler

        self._render_toolbar(handler)
        imgui.separator()
        self._render_log_content(handler)

    def _render_toolbar(self, handler):
        # Clear
        if imgui.button("Clear"):
            handler.clear()
        imgui.same_line()

        # Filter level (what to show in this window)
        imgui.set_next_item_width(90)
        changed, self._filter_idx = imgui.combo(
            "Filter##filter", self._filter_idx, _FILTER_LABELS,
        )
        if changed:
            self._filter_level = _FILTER_VALUES[self._filter_idx]
        imgui.same_line()

        # Auto-scroll
        _, self._auto_scroll = imgui.checkbox("Auto-scroll", self._auto_scroll)

        # Logger level (what gets emitted globally)
        imgui.set_next_item_width(90)
        changed, self._level_idx = imgui.combo(
            "Log level", self._level_idx, _LEVELS,
        )
        if changed:
            pylog.set_level(_LEVELS[self._level_idx].lower())

        imgui.same_line()

        # Terminal output toggle
        changed, self._term_enabled = imgui.checkbox("Terminal", self._term_enabled)
        if changed:
            if self._term_enabled and pylog.LOGGER.ch is None:
                pylog.LOGGER.ch = pylog.LOGGER.init_rich_handler(
                    level=_LEVEL_VALUES[self._level_idx],
                )
            elif not self._term_enabled and pylog.LOGGER.ch is not None:
                pylog.LOGGER.removeHandler(pylog.LOGGER.ch)
                pylog.LOGGER.ch = None

        imgui.same_line()

        # File output toggle
        changed, self._file_enabled = imgui.checkbox("File", self._file_enabled)
        if changed:
            if self._file_enabled:
                result = pfd.save_file("Log file", "farms.log", ["*.log", "*.txt"]).result()
                if result:
                    self._log_file_path = result
                    pylog.LOGGER.log2file(result)
                    self._file_enabled = True
                else:
                    self._file_enabled = False
            else:
                if pylog.LOGGER.fh is not None:
                    pylog.LOGGER.removeHandler(pylog.LOGGER.fh)
                    pylog.LOGGER.fh = None

        if self._file_enabled and self._log_file_path:
            imgui.same_line()
            imgui.text_disabled(self._log_file_path)

    def _render_log_content(self, handler):
        flags = imgui.WindowFlags_.horizontal_scrollbar
        if imgui.begin_child("scrolling", imgui.ImVec2(0, 0), imgui.ChildFlags_.none, flags):
            for level, color, msg in handler.logs:
                if level < self._filter_level:
                    continue
                rgba = _color_to_rgba(color)
                for line in msg.split('\n'):
                    if line:
                        imgui.text_colored(imgui.ImVec4(*rgba), line)

            if self._auto_scroll and imgui.get_scroll_y() >= imgui.get_scroll_max_y():
                imgui.set_scroll_here_y(1.0)
        imgui.end_child()
