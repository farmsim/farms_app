"""Viewport camera follower TaskExtension.

Dynamically adds camera-following behaviour to the MuJoCo viewport window
in farms_app.  Unlike ``farms_mujoco.simulation.extensions.CameraFollower``
(which targets ``task.viewer.cam`` from the standalone mujoco.viewer), this
extension targets the ``mj_camera`` owned by ``MuJoCoViewportWindow``.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
from farms_core.experiment.options import ExperimentOptions
from farms_core.simulation.extensions import TaskExtension

if TYPE_CHECKING:
    from dm_control.mjcf.physics import Physics


class ViewportCameraFollower(TaskExtension):
    """Follow an animat's CoM with the MuJoCo viewport camera.

    Added to / removed from ``task.extensions`` at runtime by
    ``FARMSIMExtension``.  The ``after_step`` callback smoothly interpolates
    ``camera.lookat`` toward the animat centre of mass, mirroring the logic
    in ``CameraFollower`` but operating on the viewport's ``MjvCamera``.
    """

    def __init__(
        self,
        camera: Any,
        animat_id: int = 0,
        distance: float = 1.0,
        azimuth: float = 0.0,
        elevation: float = 0.0,
        angular_velocity: float = 0.0,
    ):
        super().__init__()
        self._camera = camera
        self._animat_id = animat_id
        self._distance = distance
        self._azimuth = azimuth
        self._elevation = elevation
        self._angular_velocity = angular_velocity  # [deg/s]
        self._links: Any = None
        self._last_time = 0.0
        self._units = None

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        """Not used — instances are created directly by FARMSIMExtension."""
        raise NotImplementedError

    def initialize_episode(self, task: Any, physics: Physics):
        """Bind to the animat's link sensors and set initial camera params."""
        del physics
        self._links = task.data.animats[self._animat_id].sensors.links
        self._units = task.units
        self._last_time = 0.0
        if self._camera is not None:
            self._camera.azimuth = self._azimuth
            self._camera.distance = self._distance * self._units.meters
            self._camera.elevation = self._elevation

    def after_step(self, task: Any, physics: Physics):
        """Smoothly move camera lookat toward animat CoM."""
        if self._camera is None or self._links is None:
            return
        now = physics.time() / task.units.seconds
        time_diff, self._last_time = now - self._last_time, now
        self._camera.azimuth += self._angular_velocity * time_diff
        motion_filter = min(1.0, 10 * physics.timestep() / task.units.seconds)
        com = np.array(
            self._links.global_com_position(iteration=task.iteration - 1)
        ) * task.units.meters
        self._camera.lookat = (
            motion_filter * com + (1.0 - motion_filter) * self._camera.lookat
        )
