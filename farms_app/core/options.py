""" Options for FARMS Application """

from typing import Dict, Iterable, List

from farms_app.backends.manager import PlatformType, RendererType
from farms_core.options import Options


class BackendOptions(Options):
    """ Backend renderer options """

    def __init__(self, platform: str, renderer: str):
        super().__init__()
        self.platform = platform
        self.renderer = renderer

    @classmethod
    def defaults(cls):
        return cls(
            platform=PlatformType.GLFW.value,
            renderer=RendererType.OPENGL2.value,
        )

    @classmethod
    def from_options(cls, opts: Dict):
        return cls(
            platform=opts.get("platform", PlatformType.GLFW.value),
            renderer=opts.get("renderer", RendererType.OPENGL2.value),
        )


class WindowOptions(Options):
    """ Window options """

    def __init__(self, vsync: bool, fullscreen: bool):
        super().__init__()
        self.vsync = vsync
        self.fullscreen = fullscreen

    @classmethod
    def defaults(cls):
        return cls(vsync=True, fullscreen=False)

    @classmethod
    def from_options(cls, opts: Dict):
        return cls(
            vsync=opts.get("vsync", True),
            fullscreen=opts.get("fullscreen", False),
        )


class DockingOptions(Options):
    """ Docking Options """

    def __init__(self, enabled: bool, layout_config: str = None):
        super().__init__()
        self.enabled = enabled
        self.layout_config = layout_config

    @classmethod
    def defaults(cls):
        return cls(enabled=True, layout_config=None)

    @classmethod
    def from_options(cls, opts: Dict):
        return cls(
            enabled=opts.get("enabled", True),
            layout_config=opts.get("layout_config", None),
        )


class ExtensionOptions(Options):
    """Extension Options """

    def __init__(self, auto_enable: List[str], state: dict):
        super().__init__()
        self.auto_enable = auto_enable
        self.state = state

    @classmethod
    def defaults(cls):
        return cls(auto_enable=["status_bar"], state={})

    @classmethod
    def from_options(cls, opts: Dict):
        return cls(
            auto_enable=opts.get("auto_enable", ["status_bar"]),
            state=opts.get("state", {}),
        )


class FontOptions(Options):
    """ Font options """

    def __init__(self, name: str, size: int):
        super().__init__()
        self.name = name
        self.size = size

    @classmethod
    def defaults(cls):
        return cls(
            name="JetBrainsMono[wght].ttf",
            size=16,
        )

    @classmethod
    def from_options(cls, opts: Dict):
        return cls(
            name=opts.get("name", "JetBrainsMono[wght].ttf"),
            size=opts.get("size", 16),
        )


class ApplicationOptions(Options):
    """ Application options """

    def __init__(
            self,
            title: str,
            geometry: List[int],
            resizable: bool,
            backend: BackendOptions,
            window: WindowOptions,
            docking: DockingOptions,
            extension: ExtensionOptions,
            fonts: FontOptions,
            fps: float,
            fps_idle: float,
            enable_idling: bool,
    ):
        super().__init__()
        self.title = title
        self.geometry = geometry
        self.resizable = resizable
        self.backend = backend
        self.window = window
        self.docking = docking
        self.extension = extension
        self.fonts = fonts
        self.fps = fps
        self.fps_idle = fps_idle
        self.enable_idling = enable_idling

    @classmethod
    def defaults(cls, **kwargs):
        return cls(
            title=kwargs.pop("title", "FARMS"),
            geometry=kwargs.pop("geometry", [720, 1080]),
            resizable=kwargs.pop("resizable", True),
            backend=kwargs.pop("backend", BackendOptions.defaults()),
            window=kwargs.pop("window", WindowOptions.defaults()),
            docking=kwargs.pop("docking", DockingOptions.defaults()),
            extension=kwargs.pop("extension", ExtensionOptions.defaults()),
            fonts=kwargs.pop("fonts", FontOptions.defaults()),
            fps=kwargs.pop("fps", 60),
            fps_idle=kwargs.pop("fps_idle", 9.0),
            enable_idling=kwargs.pop("enable_idling", False),
        )

    @classmethod
    def load(cls, file_path: str):
        """Load from file, reconstructing sub-option objects."""
        opts = Options.load(file_path)
        return cls.from_options(opts)

    @classmethod
    def from_options(cls, opts: Dict):
        """Construct from a dict (e.g. parsed YAML)."""
        return cls(
            title=opts.get("title", "FARMS"),
            geometry=opts.get("geometry", [720, 1080]),
            resizable=opts.get("resizable", True),
            backend=BackendOptions.from_options(opts["backend"]) if "backend" in opts else BackendOptions.defaults(),
            window=WindowOptions.from_options(opts["window"]) if "window" in opts else WindowOptions.defaults(),
            docking=DockingOptions.from_options(opts["docking"]) if "docking" in opts else DockingOptions.defaults(),
            extension=ExtensionOptions.from_options(opts["extension"]) if "extension" in opts else ExtensionOptions.defaults(),
            fonts=FontOptions.from_options(opts["fonts"]) if "fonts" in opts else FontOptions.defaults(),
            fps=opts.get("fps", 60),
            fps_idle=opts.get("fps_idle", 9.0),
            enable_idling=opts.get("enable_idling", False),
        )
