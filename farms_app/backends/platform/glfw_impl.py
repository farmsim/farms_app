""" Generalized OpenGL 2/3 Support """


import platform
import sys
from enum import Enum
from typing import Tuple, Any
import ctypes

from ..base import BasePlatformBackend
from farms_core import pylog


if platform.system() == "Darwin":
    from imgui_bundle import imgui, implot
    # Important to import GL and glfw after imgui for cross-platform support of GL2
    import glfw
    import OpenGL.GL as GL
elif platform.system() == "Linux":
    # Important to import GL before imgui for cross-platform support of GL2
    import OpenGL.GL as GL
    import glfw
    from imgui_bundle import imgui, implot
else:
    # Important to import GL before imgui for cross-platform support of GL2
    import OpenGL.GL as GL
    import glfw
    from imgui_bundle import imgui, implot


class GLFWPlatformBackend(BasePlatformBackend):
    """GLFW platform backend """

    def __init__(self, gl_version: OpenGLVersion = OpenGLVersion.AUTO):
        self.gl_version = gl_version
        self.window = None
        self.glsl_version = None
        self._initialized = False

    def _setup_glfw_hints(self, major: int, minor: int):
        """Setup GLFW context hints"""
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, major)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, minor)

        if major >= 3 and minor >= 2:
            if platform.system() == "Darwin":
                glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
                glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)

        # Common hints
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)

    def _create_window(self, name: str, width: int, height: int) -> Any:
        """Create GLFW window with error handling"""
        glfw.set_error_callback(self._glfw_error_callback)

        if not glfw.init():
            raise pylog.error("Could not initialize GLFW")

        major, minor, self.glsl_version = self._determine_gl_version()
        self._setup_glfw_hints(major, minor)

        window = glfw.create_window(width, height, name, None, None)
        if not window:
            glfw.terminate()
            raise pylog.error(f"Could not create window with OpenGL {major}.{minor}")

        glfw.make_context_current(window)
        glfw.swap_interval(1)  # Enable vsync

        return window

    def _setup_imgui(self):
        """Setup ImGui context and backends"""
        imgui.create_context()
        _ = implot.create_context()
        io = imgui.get_io()

        # Configure ImGui
        io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard.value
        io.config_flags |= imgui.ConfigFlags_.docking_enable.value
        if platform.system() != "Linux":
            io.config_flags |= imgui.ConfigFlags_.viewports_enable.value
            io.config_viewports_no_auto_merge = True
            io.config_viewports_no_task_bar_icon = True

        # Setup style
        imgui.style_colors_dark()
        implot.style_colors_dark()

        # Viewport style adjustments
        style = imgui.get_style()
        if platform.system() != "Linux":
            if io.config_flags & imgui.ConfigFlags_.viewports_enable.value:
                style.window_rounding = 0.0
                window_bg_color = style.color_(imgui.Col_.window_bg.value)
                window_bg_color.w = 1.0
                style.set_color_(imgui.Col_.window_bg.value, window_bg_color)

        # setup backends
        window_address = ctypes.cast(self.window, ctypes.c_void_p).value
        imgui.backends.glfw_init_for_opengl(window_address, True)

        # Choose OpenGL backend based on version
        major, minor, _ = self._determine_gl_version()
        if major >= 3:
            imgui.backends.opengl3_init(self.glsl_version)
        else:
            imgui.backends.opengl2_init()

    def initialize(self, name: str = "FARMS", width: int = 1280, height: int = 720, **kwargs) -> Any:
        """Initialize GLFW backend"""
        if self._initialized:
            return self.window
        try:
            self.window = self._create_window(name, width, height)
            self._setup_imgui()
            self._initialized = True
            pylog.debug(f"GL_VERSION: {GL.glGetString(GL.GL_VERSION).decode()}")
            pylog.debug(f"GL_RENDERER: {GL.glGetString(GL.GL_RENDERER).decode()}")
            pylog.debug(f"GL_VENDOR: {GL.glGetString(GL.GL_VENDOR).decode()}")
            return self.window
        except Exception as e:
            self.cleanup()
            raise pylog.error(f"Backend initialization failed: {e}")

    def cleanup(self):
        """Cleanup GLFW resources"""
        if self._initialized:
            major, minor, _ = self._determine_gl_version()
            if major >= 3:
                imgui.backends.opengl3_shutdown()
            else:
                imgui.backends.opengl2_shutdown()
            imgui.backends.glfw_shutdown()
            imgui.destroy_context()

        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()
        self._initialized = False

    def begin_frame(self):
        """Begin frame rendering"""
        glfw.poll_events()

        major, minor, _ = self._determine_gl_version()
        if major >= 3:
            imgui.backends.opengl3_new_frame()
        else:
            imgui.backends.opengl2_new_frame()
        imgui.backends.glfw_new_frame()
        imgui.new_frame()
        imgui.dock_space_over_viewport(viewport=imgui.get_main_viewport())

    def end_frame(self):
        """End frame rendering"""
        io = imgui.get_io()

        imgui.render()
        width, height = glfw.get_framebuffer_size(self.window)
        GL.glViewport(0, 0, width, height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        major, minor, _ = self._determine_gl_version()
        if major >= 3:
            imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())
        else:
            imgui.backends.opengl2_render_draw_data(imgui.get_draw_data())

        # Multi-viewport support
        if platform.system() != "Linux":
            if io.config_flags & imgui.ConfigFlags_.viewports_enable.value:
                backup_current_context = glfw.get_current_context()
                imgui.update_platform_windows()
                imgui.render_platform_windows_default()
                glfw.make_context_current(backup_current_context)

        glfw.swap_buffers(self.window)

    def draw(self, data):
        """ Draw imgui data """
        if self.gl_version == OpenGLVersion.GL2:
            imgui.backends.opengl2_render_draw_data(data)
        elif self.gl_version == OpenGLVersion.GL3:
            imgui.backends.opengl3_render_draw_data(data)

    def should_close(self) -> bool:
        """Check if window should close"""
        return glfw.window_should_close(self.window) if self.window else True

    def poll_events(self):
        """ Poll GLFW events """
        glfw.poll_events()

    @staticmethod
    def _glfw_error_callback(error: int, description: str) -> None:
        sys.stderr.write(f"GLFW Error {error}: {description}\n")


# Convenience functions for backward compatibility
def create_window(name: str = "FARMS", width: int = 1280, height: int = 720,
                 gl_version: OpenGLVersion = OpenGLVersion.AUTO):
    """Create window with specified OpenGL version"""
    backend = GLFWBackend(gl_version)
    return backend.initialize(name, width, height)


def initialize(name: str = "FARMS", width: int = 1280, height: int = 720,
              gl_version: OpenGLVersion = OpenGLVersion.AUTO):
    """Initialize backend (backward compatibility)"""
    backend = GLFWBackend(gl_version)
    window = backend.initialize(name, width, height)
    return window, backend
