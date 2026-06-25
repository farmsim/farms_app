""" Base interface for backend """
# DO NOT IMPORT IMGUI here! Breaks cross-platform support for GL2 :(


from abc import ABC, abstractmethod
from typing import Any


class BackendError(Exception):
    """Backend-specific errors"""
    pass


class BaseBackend(ABC):
    """Abstract base class for all backends"""

    platform_name: str = "Unknown"
    renderer_name: str = "Unknown"
    _screenshot_path: "str | None" = None

    @abstractmethod
    def initialize(self, name: str, width: int, height: int, **kwargs) -> Any:
        """Initialize the backend and return window handle"""

    @abstractmethod
    def cleanup(self):
        """Cleanup backend resources"""

    @abstractmethod
    def begin_frame(self):
        """Begin frame rendering"""

    @abstractmethod
    def end_frame(self):
        """End frame rendering"""

    @abstractmethod
    def draw(self, data: "imgui.ImDrawData"):
        """ valid after Render() and until the next call to NewFrame(). this is what you have to render. """

    @abstractmethod
    def should_close(self) -> bool:
        """Check if window should close"""

    @abstractmethod
    def request_close(self):
        """Request the window to close"""

    @abstractmethod
    def cancel_close(self):
        """Cancel a pending close request."""

    @abstractmethod
    def poll_events(self):
         """Poll events"""

    @abstractmethod
    def event_timeout(self, timeout_seconds):
        """ Event timeout """

    def _do_screenshot(self, path: str):
        """Save the current framebuffer to path. Override in backends that support pixel readback."""


class BaseRendererBackend(ABC):
    """Abstract base class for rendering backends"""

    @abstractmethod
    def initialize(self, name: str, width: int, height: int, **kwargs) -> Any:
        """Initialize the backend and return window handle"""

    @abstractmethod
    def cleanup(self):
        """Cleanup backend resources"""

    @abstractmethod
    def begin_frame(self):
        """Begin frame rendering"""

    @abstractmethod
    def end_frame(self):
        """End frame rendering"""

    @abstractmethod
    def draw(self, data: "imgui.ImDrawData"):
        """ valid after Render() and until the next call to NewFrame(). this is what you have to render. """

    @abstractmethod
    def should_close(self) -> bool:
        """Check if window should close"""

    @abstractmethod
    def poll_events(self):
         """Poll events"""


class BasePlatformBackend(ABC):
    """ Abstract base class for platform backends """

    @abstractmethod
    def initialize(self, name: str, width: int, height: int, **kwargs) -> Any:
        """Initialize the backend and return window handle"""

    @abstractmethod
    def create_window(self, name: str, width: int, height: int, **kwargs) -> bool:
        """ Create a window """

    @abstractmethod
    def cleanup(self):
        """Cleanup backend resources"""

    @abstractmethod
    def swap_buffers(self):
        """Cleanup backend resources"""

    @abstractmethod
    def should_close(self) -> bool:
        """Check if window should close"""

    @abstractmethod
    def poll_events(self):
         """Poll events"""
