""" Main application level menus """

import datetime
import sys
from importlib.metadata import version

import farms_app
from farms_core import pylog
from imgui_bundle import imgui, portable_file_dialogs as pfd


_ABOUT_ROWS = (
    ("VERSION", farms_app.__version__),
    ("PYTHON", sys.version.split()[0]),
    ("IMGUI_BUNDLE", version("imgui-bundle")),
    ("LICENSE", "Apache-2.0"),
)


_COL0_WIDTH = 200.0

def _info_table(table_id, rows):
    """Render a two-column label/value table."""
    if imgui.begin_table(table_id, 2, imgui.TableFlags_.borders_inner_v.value):
        imgui.table_setup_column("##label", imgui.TableColumnFlags_.width_fixed.value, _COL0_WIDTH)
        imgui.table_setup_column("##value")
        for label, value in rows:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(label)
            imgui.table_next_column()
            imgui.text(value)
        imgui.end_table()


def render_main_menu(app):
    """Render the main menu bar.

    Args:
        app: FARMSApplication instance.
    """
    imgui.begin_main_menu_bar()

    # File
    if imgui.begin_menu("File"):
        if imgui.menu_item_simple("Screenshot"):
            default = datetime.datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.png")
            result = pfd.save_file("Save screenshot", default, ["*.png"]).result()
            if result:
                app._screenshot_path = result
        if imgui.menu_item_simple("Restore Defaults"):
            app.extension_manager.reset_all_state()
        imgui.separator()
        if imgui.menu_item_simple("Quit", shortcut="Alt+Q"):
            app.backend.request_close()
        imgui.end_menu()

    # View
    if imgui.begin_menu("View"):
        if imgui.begin_menu("Theme"):
            imgui.show_style_selector("Styles")
            imgui.show_style_editor()
            imgui.end_menu()
        imgui.separator()
        if imgui.begin_menu("Extensions"):
            for name, extension in app.extension_manager._enabled_exts.items():
                clicked, new_state = imgui.menu_item(
                    name, shortcut="", p_selected=not extension.obj.hide
                )
                if clicked:
                    extension.obj.hide = not new_state
            imgui.end_menu()
        imgui.end_menu()

    # Extensions
    if imgui.begin_menu("Extensions"):
        for name in app.extension_manager.names:
            clicked, new_state = imgui.menu_item(
                name, shortcut="",
                p_selected=name in app.extension_manager._enabled_exts,
            )
            if clicked and new_state:
                app.extension_manager.enable(name)
            elif clicked and not new_state:
                app.extension_manager.disable(name)
        imgui.end_menu()

    # Debug
    if imgui.begin_menu("Debug"):
        clicked, new_state = imgui.menu_item(
            "Show Metrics", shortcut="", p_selected=app.show_metrics_window
        )
        if clicked:
            app.show_metrics_window = new_state
        clicked, new_state = imgui.menu_item(
            "Frame Timer", shortcut="", p_selected=app.frame_timer.enabled
        )
        if clicked:
            app.frame_timer.enabled = new_state
        if imgui.begin_menu("Level"):
            _curr_level = pylog.get_level()
            for level in ("debug", "info", "warning", "error", "critical"):
                if imgui.menu_item(
                    level, shortcut="",
                    p_selected=(_curr_level == level),
                )[0]:
                    pylog.set_level(level)
            imgui.end_menu()
        imgui.end_menu()

    # Help
    if imgui.begin_menu("Help"):
        imgui.separator_text("Graphics")
        _info_table("##graphics", (
            ("PLATFORM", app.backend.platform_name),
            ("RENDERER", app.backend.renderer_name),
        ))
        imgui.separator()
        _info_table("##about", _ABOUT_ROWS)
        imgui.end_menu()

    # Extension-contributed top-level menus
    for name, extension in app.extension_manager._enabled_exts.items():
        extension.obj.menu()

    imgui.end_main_menu_bar()
