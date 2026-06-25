""" Extensions management and implementation """

import inspect

from farms_app.console import console
from farms_app.core.hooks import Hooks
from farms_app.core.window import Window
from farms_core import pylog
from imgui_bundle import imgui
from stevedore import EnabledExtensionManager


EXTENSION_NAMESPACE = "farms.app.extension"


###########
# Manager #
###########
class ExtensionManager:
    """Manager class for all App Extensions under the namespace farms.app

    Responsibilities:
    - Plugin discovery from multiple sources
    - Dependency resolution and loading order
    - Runtime enable/disable
    """

    def __init__(self, fail_on_load=True):
        super().__init__()
        self.fail_on_load = fail_on_load
        self._enabled_exts: dict[str, EnabledExtension] = {}
        self.frame_timer = None  # set by FARMSApplication
        self.dockspace_id: int = 0  # set by FARMSApplication
        self._saved_state: dict = {}  # set by FARMSApplication

        self._mgr = EnabledExtensionManager(
            namespace=EXTENSION_NAMESPACE,
            check_func=self.check_cb,
            on_load_failure_callback=self.load_error_cb,
        )

    def load_error_cb(self, manager, entry_point, exception):
        pylog.error(f"Could not load extension {entry_point} by {manager}")
        if self.fail_on_load:
            console.print_exception()

    def check_cb(self, ext):
        return (
            inspect.isclass(ext.plugin) and issubclass(ext.plugin, Extension)
        )

    @property
    def names(self):
        """All discovered extension names (enabled and disabled)."""
        return self._mgr.names()

    @property
    def enabled(self):
        """List of (name, extension) pairs for all enabled extensions."""
        return [(name, ee.obj) for name, ee in self._enabled_exts.items()]

    def is_enabled(self, name: str) -> bool:
        """Check if an extension is currently enabled."""
        return name in self._enabled_exts

    def get(self, name: str):
        """Get an enabled extension instance by name."""
        if name not in self._enabled_exts:
            return None
        return self._enabled_exts[name].obj

    def enable(self, name: str) -> bool:
        """Instantiate and enable an extension."""
        if name in self._enabled_exts:
            pylog.warning(f"Extension {name} is already enabled")
            return True

        if name not in self.names:
            pylog.error(f"Requested extension {name} is not available")
            return False

        try:
            _ext = self._mgr[name]
            ext_obj = _ext.plugin()
            ext_obj.dockspace_id = self.dockspace_id
            self._enabled_exts[name] = EnabledExtension(
                entry_point=_ext.entry_point,
                obj=ext_obj,
            )
            ext_obj.on_enable()
            if name in self._saved_state:
                ext_obj.on_restore_state(self._saved_state[name])
                pylog.info(f"Restored state for {name}")
            pylog.info(f"Enabled extension {name}")
            return True
        except Exception as e:
            pylog.error(f"Failed enabling extension {name} with error: {e}")
            console.print_exception(show_locals=True)
            return False

    def disable(self, name: str) -> bool:
        """Disable an extension, calling lifecycle hooks."""
        if name not in self._enabled_exts:
            pylog.error(f"Requested extension {name} is not enabled")
            return False

        try:
            ext_obj = self._enabled_exts[name].obj
            ext_obj.on_disable()
            ext_obj.cleanup()
            del self._enabled_exts[name]
            pylog.info(f"Disabled extension {name}")
            return True
        except Exception as e:
            pylog.error(f"Error disabling extension {name}: {e}")
            console.print_exception(show_locals=True)
            return False

    def tick(self, dt: float):
        """Per-frame dispatch: update -> event -> render for all enabled extensions."""
        ft = self.frame_timer
        for name, enabled_ext in list(self._enabled_exts.items()):
            try:
                if ft:
                    ft.begin_scope(name)
                    ft.begin_phase("update")
                enabled_ext.obj.pre_update(dt)
                enabled_ext.obj.on_update(dt)
                enabled_ext.obj.post_update(dt)
                if ft:
                    ft.end_phase("update")
                    ft.begin_phase("event")
                enabled_ext.obj.on_event()
                if ft:
                    ft.end_phase("event")
                    ft.begin_phase("render")
                enabled_ext.obj.render(ft)
                if ft:
                    ft.end_phase("render")
                    ft.end_scope()
            except Exception as e:
                console.print_exception()
                pylog.error(f"Error in extension {name}: {e}")

    def save_state(self) -> dict:
        """Collect state from all enabled extensions. Returns a dict."""
        state = {}
        for name, ee in self._enabled_exts.items():
            try:
                ext_state = ee.obj.on_save_state()
                if ext_state:
                    state[name] = ext_state
            except Exception as e:
                pylog.error(f"Error saving state for {name}: {e}")
        return state

    def reset_all_state(self):
        """Reset state for all enabled extensions to defaults."""
        self._saved_state = {}
        for name, ee in self._enabled_exts.items():
            try:
                ee.obj.on_reset_state()
                pylog.info(f"Reset state for {name}")
            except Exception as e:
                pylog.error(f"Error resetting state for {name}: {e}")

    def shutdown(self):
        """Disable and cleanup all enabled extensions. Called on app exit."""
        for name in list(self._enabled_exts):
            self.disable(name)

    def unload(self, name: str) -> bool:
        """Unload an extension completely (disable + remove from cache)."""
        if name not in self._mgr:
            pylog.error(f"Unknown extension {name} cannot be unloaded")
            return False

        self.disable(name)
        for index, entry_point in list(self._mgr.ENTRY_POINT_CACHE[EXTENSION_NAMESPACE]):
            if name == entry_point.name:
                break
        self._mgr.ENTRY_POINT_CACHE[EXTENSION_NAMESPACE].pop(index)
        pylog.debug(f"Removing extension {name} from loaded extensions")
        return True


class EnabledExtension:
    """Container for an enabled extension."""

    def __init__(self, entry_point: str, obj: 'Extension'):
        self.entry_point: str = entry_point
        self.obj: 'Extension' = obj


##############
# Extensions #
##############
class Extension:
    """Base class for all extensions.

    Lifecycle:
        on_enable() -> [per frame: on_update(dt) -> on_event() -> render()] -> on_disable() -> cleanup()

    Override on_update(dt) for simulation stepping (decoupled from frame rate).
    Override on_event() for input handling.
    Override on_render() for extension-level drawing.
    Override render() only if you need full control over window iteration.
    """

    category = "custom"

    def __init__(self, name: str):
        self.name = name
        self.hide: bool = False
        self.dockspace_id: int = 0
        self.windows: dict[str, Window] = {}
        self.hooks = Hooks("pre_update", "post_update")

    ###########
    # Windows #
    ###########
    def register_window(self, window: Window):
        """Register a window with this extension."""
        self.windows[window.name] = window

    def unregister_window(self, window: Window):
        """Unregister a window from this extension."""
        self.windows.pop(window.name, None)

    def init_windows(self):
        """Initialize all uninitialized windows. Call when data is ready."""
        for window in self.windows.values():
            if not window._initialized:
                window.initialize()

    def show_all_windows(self):
        """Show all windows for this extension."""
        for window in self.windows.values():
            window.show()

    def hide_all_windows(self):
        """Hide all windows for this extension."""
        for window in self.windows.values():
            window.hide()

    #############
    # Lifecycle #
    #############
    def on_enable(self):
        """Called when extension is enabled."""

    def on_disable(self):
        """Called when extension is about to be disabled."""

    def pre_update(self, dt: float):
        """Called before on_update. Fires pre_update hooks."""
        self.hooks["pre_update"].fire(self, dt)

    def on_update(self, dt: float):
        """Called once per frame with frame delta time.
        Simulation stepping goes here. The extension decides how many
        steps to run based on dt — the framework does not own the
        simulation clock.
        """

    def post_update(self, dt: float):
        """Called after on_update. Fires post_update hooks."""
        self.hooks["post_update"].fire(self, dt)

    def on_event(self):
        """Called once per frame for input handling."""

    def menu(self):
        """Called inside the main menu bar. Extension renders its own
        namespaced top-level menu here (e.g., begin_menu("MyExtension")).
        """

    def on_render(self):
        """Called during every render cycle for extension-level rendering."""

    def render(self, frame_timer=None):
        """Render this extension and its windows.
        Override only if you need custom control over window iteration.
        """
        if self.hide:
            return

        self.on_render()

        for window in self.windows.values():
            if window._initialized:
                if frame_timer:
                    frame_timer.begin_window(window.name)
                window._render()
                if frame_timer:
                    frame_timer.end_window(window.name)

    def cleanup(self):
        """Clean up resources before shutdown."""

    def on_save_state(self) -> dict:
        """Return serializable state to persist across runs. Override to save custom state."""
        return {}

    def on_restore_state(self, state: dict):
        """Restore state from a previous run. Override to load custom state."""

    def on_reset_state(self):
        """Reset extension state to defaults. Override to handle cleanup."""

    def dependencies(self):
        """Return list of extension names this depends on."""
        return []

    ############
    # Metadata #
    ############
    def get_info(self) -> dict:
        """Return extension metadata."""
        return {
            "name": self.name,
            "windows": self.windows,
            "hidden": self.hide,
        }

    ###########
    # Utility #
    ###########
    def create_window_id(self, window_name: str) -> str:
        """Create a unique window identifier for this extension."""
        return f"{window_name}##{self.name}"


class UIExtension(Extension):
    """Extensions for app-level UI chrome (status bar, toolbars, debug panels).

    Isolated from regular extensions. Future: always enabled,
    rendered separately, no dockspace.

    UI extensions render directly via on_render() — they don't
    participate in the window system.
    """

    category = "ui"

    def render(self, frame_timer=None):
        """UI extensions render directly — no window iteration."""
        if self.hide:
            return
        self.on_render()
