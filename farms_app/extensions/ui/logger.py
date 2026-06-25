""" Logger """

from colorama import Fore
from farms_core import pylog
from imgui_bundle import imgui

from farms_app.core.extension import UIExtension
from farms_app.core.window import BaseWindow


ANSI_RGB_MAP = {
    Fore.CYAN: (0, 1.0, 1.0, 1.0),
    Fore.GREEN: (0, 1.0, 0, 1.0),
    Fore.YELLOW: (1.0, 1.0, 0, 1.0),
    Fore.RED: (1.0, 0, 0, 1.0),
    Fore.MAGENTA: (1.0, 0, 1.0, 1.0),
}


class DebugExtension(UIExtension):
    """ Logger """

    def __init__(self):
        name = "Debug"
        super().__init__(name=name)
        self.register_window(LoggerWindow(self))

    def get_dependencies(self):
        """ Get extension dependencies """

    def cleanup(self):
        """ Cleanup resources before unloading the extension """


class LoggerWindow(BaseWindow):
    """ Render Status Bar """

    def __init__(self, extension) -> None:
        name: str = "log"
        window_flags = (
            imgui.WindowFlags_.no_title_bar |
            imgui.WindowFlags_.no_resize |
            imgui.WindowFlags_.no_move |
            imgui.WindowFlags_.no_scrollbar |
            imgui.WindowFlags_.no_collapse |
            imgui.WindowFlags_.no_scroll_with_mouse |
            imgui.WindowFlags_.no_bring_to_front_on_focus |
            imgui.WindowFlags_.no_nav_focus
        )
        super().__init__(
            name=name,
            extension=extension,
            visible=True,
            dock_to_extension=False
        )

    def on_initialize(self):
        """ On initialize """

    def on_update(self):
        """ On update """

    def on_render(self):
        """ Render main extension dockspace """
        if imgui.begin_popup("Options"):
            imgui.checkbox("Auto-scroll", True)
            imgui.end_popup()

        # Main window
        if imgui.button("Options"):
            imgui.open_popup("Options")
        imgui.same_line()
        clear = imgui.button("Clear")
        imgui.same_line()
        copy = imgui.button("Copy")

        if imgui.begin_child("scrolling", imgui.ImVec2((0, 0)), imgui.ChildFlags_.none, imgui.WindowFlags_.horizontal_scrollbar):
            # for color, line in pylog.LOGGER.get_gui_logs():
            #      imgui.text_colored(imgui.ImVec4(*ANSI_RGB_MAP.get(color)), line)
            imgui.text("Hello world!")
        imgui.end_child()
