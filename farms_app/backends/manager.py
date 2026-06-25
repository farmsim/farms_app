from enum import Enum
from typing import Optional

from farms_core import pylog

from .base import BaseBackend
from .glfw_impl import GLFWBackend


class PlatformType(str, Enum):
    """ Type of Platform """
    GLFW = 'glfw'


class RendererType(str, Enum):
    """ Type of Renderer """
    OPENGL2 = 'gl2'
    OPENGL3 = 'gl3'


class BackendManager:
    """Manages different backend implementations"""

    def __init__(self) -> None:
        self.current_backend: Optional[BaseBackend] = None
        self._available_backends = {
            PlatformType.GLFW: GLFWBackend,
        }

    def create_backend(self, backend_type: str = 'glfw', **kwargs) -> BaseBackend:
        """Create a backend instance"""
        if backend_type not in [member.value for member in PlatformType]:
            pylog.error(f"Unknown backend: {backend_type}")
            raise ValueError

        backend_class = self._available_backends[PlatformType(backend_type)]
        return backend_class(**kwargs)

    def initialize(self, backend_type: str = 'glfw', **kwargs) -> BaseBackend:
        """Initialize and set current backend"""
        if self.current_backend:
            self.current_backend.cleanup()

        self.current_backend = self.create_backend(backend_type, **kwargs)
        return self.current_backend

    def cleanup(self):
        """Cleanup current backend"""
        if self.current_backend:
            self.current_backend.cleanup()
            self.current_backend = None
