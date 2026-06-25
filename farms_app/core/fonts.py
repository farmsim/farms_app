""" Managing and handling fonts """

from imgui_bundle import hello_imgui

from farms_app.utils import paths

from .options import FontOptions


def load_fonts(options: FontOptions) -> None:
    """Load primary font first (becomes default), then all others in the fonts directory."""
    fonts_dir = paths.get_project_root().joinpath("farms_app", "assets", "fonts")
    hello_imgui.load_font_ttf_with_font_awesome_icons(
        str(fonts_dir.joinpath(options.name)), options.size
    )
    for ttf in sorted(fonts_dir.glob("*.ttf")):
        if ttf.name != options.name:
            hello_imgui.load_font_ttf_with_font_awesome_icons(str(ttf), options.size)
