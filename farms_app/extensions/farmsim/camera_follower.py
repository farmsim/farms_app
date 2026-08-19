"""Viewport trail viewer TaskExtensions.

Provides ``ViewportCameraFollower``, ``ViewportTrailCoMViewer``, and
``ViewportCoMViewer`` — farms_app equivalents of the ``CameraFollower``,
``TrailCoMViewer``, and ``CoMViewer`` classes in
``farms_mujoco.simulation.extensions``, adapted to target the ``mj_camera``
and ``mj_scene`` owned by ``MuJoCoViewportWindow`` instead of the standalone
``mujoco.viewer`` handle (which is ``None`` in the app).
"""

import mujoco
import numpy as np

from farms_core.experiment.options import ExperimentOptions
from farms_core.simulation.extensions import TaskExtension

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
        camera,
        animat_id: int = 0,
        distance: float = 1.0,
        azimuth: float = 0.0,
        elevation: float = 0.0,
        angular_velocity: float = 0.0,
    ):
        super().__init__()
        self.camera = camera
        self.animat_id = animat_id
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        self.angular_velocity = angular_velocity  # [deg/s]
        self.links = None
        self.last_time = 0.0
        self.units = None

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        """Not used — instances are created directly by FARMSIMExtension."""
        raise NotImplementedError

    def initialize_episode(self, task, physics: Physics):
        """Bind to the animat's link sensors and set initial camera params."""
        del physics
        self.links = task.data.animats[self.animat_id].sensors.links
        self.units = task.units
        self.last_time = 0.0
        if self.camera is not None:
            self.camera.azimuth = self.azimuth
            self.camera.distance = self.distance * self.units.meters
            self.camera.elevation = self.elevation

    def after_step(self, task, physics: Physics):
        """Smoothly move camera lookat toward animat CoM."""
        if self.camera is None or self.links is None:
            return
        now = physics.time() / task.units.seconds
        time_diff, self.last_time = now - self.last_time, now
        self.camera.azimuth += self.angular_velocity * time_diff
        motion_filter = min(1.0, 10 * physics.timestep() / task.units.seconds)
        com = np.array(
            self.links.global_com_position(iteration=task.iteration - 1)
        ) * task.units.meters
        self.camera.lookat = (
            motion_filter * com + (1.0 - motion_filter) * self.camera.lookat
        )


class ViewportTrailCoMViewer(TaskExtension):
    """Draw a trail of the animat's CoM in the MuJoCo viewport scene.

    Like ``TrailCoMViewer`` from ``farms_mujoco``, but instead of adding
    geoms to ``viewer.user_scn``, the trail segments are stored and drawn
    into the viewport's ``mj_scene`` by ``MuJoCoViewportWindow.render_mujoco``
    after ``mjv_updateScene`` and before ``mjr_render``.
    """

    def __init__(
        self,
        animat_id: int = 0,
        spacing: int = 10,
        width: int = 5,
        rgba: list[float] | None = None,
    ):
        super().__init__()
        self.animat_id = animat_id
        self.spacing = spacing
        self.width = width
        self.rgba = rgba or [1.0, 0.3, 0.0, 0.7]
        self.links = None
        self.units = None
        self.segments: list[tuple[np.ndarray, np.ndarray]] = []
        self.pos_old: np.ndarray | None = None
        self.pos_new: np.ndarray | None = None

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        """Not used — instances are created directly by FARMSIMExtension."""
        raise NotImplementedError

    def initialize_episode(self, task, physics: Physics):
        """Bind to the animat's link sensors."""
        del physics
        self.links = task.data.animats[self.animat_id].sensors.links
        self.units = task.units
        self.segments = []
        self.pos_new = self.pos_old = np.array(
            self.links.global_com_position(0)
        ) * self.units.meters

    def after_step(self, task, physics: Physics):
        """Record a new trail segment every ``spacing`` iterations."""
        del physics
        if self.links is None:
            return
        iteration = task.iteration - 1
        if not iteration % self.spacing:
            self.pos_new = np.array(
                self.links.global_com_position(iteration)
            ) * self.units.meters
            if self.pos_old is not None:
                self.segments.append((self.pos_old.copy(), self.pos_new.copy()))
            self.pos_old = self.pos_new

    def render_trail(self, scene: mujoco.MjvScene):
        """Add trail line geoms to the scene.

        Called by ``MuJoCoViewportWindow.render_mujoco`` after
        ``mjv_updateScene`` and before ``mjr_render``.
        """
        for begin, end in self.segments:
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom=geom,
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=[1.0, 1.0, 1.0],
                pos=begin,
                mat=np.eye(3).ravel(),
                rgba=self.rgba,
            )
            mujoco.mjv_connector(
                geom=geom,
                type=mujoco.mjtGeom.mjGEOM_LINE,
                width=self.width,
                from_=begin,
                to=end,
            )
            scene.ngeom += 1


class ViewportCoMViewer(TaskExtension):
    """Draw a sphere at the animat's CoM in the MuJoCo viewport scene.

    Like ``CoMViewer`` from ``farms_mujoco``, but instead of adding a
    geom to ``viewer.user_scn``, the sphere is drawn into the viewport's
    ``mj_scene`` by ``MuJoCoViewportWindow.render_mujoco`` after
    ``mjv_updateScene`` and before ``mjr_render``.
    """

    def __init__(
        self,
        animat_id: int = 0,
        size: list[float] | None = None,
        rgba: list[float] | None = None,
    ):
        super().__init__()
        self.animat_id = animat_id
        self.size = size or [0.01, 0.0, 0.0]
        self.rgba = rgba or [1.0, 1.0, 1.0, 0.3]
        self.links = None
        self.units = None
        self.com: np.ndarray | None = None

    @classmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        """Not used — instances are created directly by FARMSIMExtension."""
        raise NotImplementedError

    def initialize_episode(self, task, physics: Physics):
        """Bind to the animat's link sensors and compute sphere radius."""
        del physics
        self.links = task.data.animats[self.animat_id].sensors.links
        self.units = task.units
        mass = np.sum(self.links.masses)
        if mass is not None:
            radius = 0.2 * ((3 * mass / 1000) / np.pi) ** (1 / 3)
            self.size = [radius, 0.0, 0.0]
        self.com = np.array(
            self.links.global_com_position(0)
        ) * self.units.meters

    def after_step(self, task, physics: Physics):
        """Update the stored CoM position."""
        del physics
        if self.links is None:
            return
        self.com = np.array(
            self.links.global_com_position(task.iteration - 1)
        ) * self.units.meters

    def render_com(self, scene: mujoco.MjvScene):
        """Add a CoM sphere geom to the scene.

        Called by ``MuJoCoViewportWindow.render_mujoco`` after
        ``mjv_updateScene`` and before ``mjr_render``.
        """
        if self.com is None or scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom=geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=self.size,
            pos=self.com,
            mat=np.eye(3).ravel(),
            rgba=self.rgba,
        )
        scene.ngeom += 1
