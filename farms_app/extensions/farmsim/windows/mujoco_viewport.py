""" MuJoCo 3D viewport window """

import mujoco
import numpy as np
from imgui_bundle import imgui
from typing import TYPE_CHECKING

from farms_core import pylog

from ....core.widget import PlaybackState
from ....core.window import Window
from ....utils.mujoco import (
    setup_scene,
    mouse_interactions,
    keyboard_interactions,
)

if TYPE_CHECKING:
    from ..extension import FARMSIMExtension


class MuJoCoViewportWindow(Window["FARMSIMExtension"]):
    """Renders MuJoCo scene to an offscreen framebuffer displayed via ImGui.

    Accesses model/data through self._extension.sim — no stale references
    when the simulation is reloaded.
    """

    def __init__(self, extension, width: int = 1280, height: int = 720):
        super().__init__("MuJoCo", extension)
        self._io = imgui.get_io()
        self.width = width
        self.height = height

        self.mj_camera = None
        self.mj_option = None
        self.mj_perturb = None
        self.mj_scene = None
        self.mj_context = None
        self.mj_viewport = None

        self._resolve_fbo = 0
        self._resolve_tex = 0
        self._needs_resolve_fbo = False

        self._viewer_pos = None
        self._viewer_size = None
        self.is_scene_hovered = False

    @property
    def model(self):
        return self._extension.sim.physics.model._model

    @property
    def data(self):
        return self._extension.sim.physics.data._data

    def on_initialize(self):
        # First, completely clean up any existing MuJoCo objects to avoid
        # caching issues
        self._cleanup_mujoco_resources()

        render_flags = {
            mujoco.mjtRndFlag.mjRND_SKYBOX: True,
            mujoco.mjtRndFlag.mjRND_REFLECTION: True,
            mujoco.mjtRndFlag.mjRND_SHADOW: True,
        }
        (self.mj_camera, self.mj_option, self.mj_perturb,
         self.mj_context, self.mj_scene, self.mj_viewport) = setup_scene(
            self.model, self.width, self.height, render_flags=render_flags,
        )
        self._needs_resolve_fbo = True

    def _cleanup_mujoco_resources(self):
        """Completely clean up all MuJoCo visualization resources."""
        try:
            # Clean up OpenGL resources
            if hasattr(self, '_resolve_fbo') and self._resolve_fbo:
                import OpenGL.GL as GL
                GL.glDeleteFramebuffers(1, [self._resolve_fbo])
                self._resolve_fbo = 0
            if hasattr(self, '_resolve_tex') and self._resolve_tex:
                import OpenGL.GL as GL
                GL.glDeleteTextures([self._resolve_tex])
                self._resolve_tex = 0

            # Clean up MuJoCo objects - set to None to allow garbage collection
            # MuJoCo objects will be automatically cleaned up when no references
            # remain
            self.mj_scene = None
            self.mj_context = None
            self.mj_camera = None
            self.mj_option = None
            self.mj_perturb = None
            self.mj_viewport = None

            pylog.debug("MuJoCo viewport resources completely cleaned up")
        except Exception as e:
            pylog.warning(f"Failed to clean up MuJoCo viewport resources: {e}")

    def _create_resolve_fbo(self, width: int, height: int):
        """Create a simple FBO with a texture attachment for ImGui display."""
        import OpenGL
        import OpenGL.GL as GL
        _prev = OpenGL.ERROR_CHECKING
        OpenGL.ERROR_CHECKING = False
        try:
            if self._resolve_fbo:
                GL.glDeleteFramebuffers(1, [self._resolve_fbo])
                GL.glDeleteTextures([self._resolve_tex])

            self._resolve_tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._resolve_tex)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8,
                width, height, 0,
                GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None,
            )
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

            self._resolve_fbo = GL.glGenFramebuffers(1)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._resolve_fbo)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                GL.GL_TEXTURE_2D, self._resolve_tex, 0,
            )
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        finally:
            OpenGL.ERROR_CHECKING = _prev
        self._needs_resolve_fbo = False

    def render_mujoco(self):
        """Render MuJoCo scene to offscreen FBO, blit to texture. Called before ImGui frame."""
        import OpenGL.GL as GL
        if self.mj_context is None:
            return
        if self._needs_resolve_fbo:
            mujoco.mjr_resizeOffscreen(self.width, self.height, self.mj_context)
            self._create_resolve_fbo(self.mj_context.offWidth, self.mj_context.offHeight)

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.mj_context)

        mujoco.mjv_updateScene(
            self.model, self.data,
            self.mj_option, self.mj_perturb, self.mj_camera,
            mujoco.mjtCatBit.mjCAT_ALL, self.mj_scene,
        )

        # Overlay extension geoms (trails, CoM spheres, etc.)
        ext = self._extension
        if ext.task is not None:
            for extension in ext.task.extensions:
                if hasattr(extension, 'render_trail'):
                    extension.render_trail(self.mj_scene)
                elif hasattr(extension, 'render_com'):
                    extension.render_com(self.mj_scene)

        mujoco.mjr_render(self.mj_viewport, self.mj_scene, self.mj_context)

        # Flush stale GL errors from MuJoCo's legacy rendering (C call, bypasses PyOpenGL)
        while mujoco.mjr_getError():
            pass

        # Blit from MuJoCo's offscreen renderbuffer to our texture FBO
        off_w = self.mj_context.offWidth
        off_h = self.mj_context.offHeight
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, int(self.mj_context.offFBO))
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, int(self._resolve_fbo))
        GL.glBlitFramebuffer(
            0, 0, off_w, off_h,
            0, 0, off_w, off_h,
            GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST,
        )
        # Force alpha=1 (MuJoCo clears to alpha=0, ImGui blends as transparent)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(self._resolve_fbo))
        GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_TRUE)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def on_render(self):
        ext = self._extension
        scrub_min, scrub_max = 0, 0
        if ext.sim is not None:
            task = ext.sim.task
            scrub_min = -min(task.iteration, task.buffer_size)
            scrub_max = 0

        ext.toolbar.render(
            playback_state=ext.playback_state,
            speed=ext.playback_speed,
            recording=ext._recording,
            scrub_value=ext._view_offset,
            scrub_min=scrub_min,
            scrub_max=scrub_max,
        )

        # Camera follow toggle
        if ext.sim is not None:
            changed, following = imgui.checkbox(
                "Follow Animat", ext._camera_follower is not None,
            )
            if changed:
                if following:
                    ext.enable_camera_follow()
                else:
                    ext.disable_camera_follow()
            imgui.same_line()
            changed, trail = imgui.checkbox(
                "CoM Trail", ext._trail_viewer is not None,
            )
            if changed:
                if trail:
                    ext.enable_trail()
                else:
                    ext.disable_trail()
            imgui.same_line()
            changed, com = imgui.checkbox(
                "CoM Viewer", ext._com_viewer is not None,
            )
            if changed:
                if com:
                    ext.enable_com_view()
                else:
                    ext.disable_com_view()

        if ext.sim is None:
            imgui.text("No simulation loaded")
            return

        self._render_scene()

    def _resize(self, width: int, height: int):
        """Mark resize needed (actual resize happens in pre-frame)."""
        self.width = width
        self.height = height
        self.mj_viewport.width = width
        self.mj_viewport.height = height
        self._needs_resolve_fbo = True

    def _render_scene(self):
        """Display pre-rendered MuJoCo texture as ImGui image."""
        self._viewer_pos = imgui.get_cursor_screen_pos()

        avail_w, avail_h = imgui.get_content_region_avail()
        if avail_w <= 0 or avail_h <= 0:
            return

        new_w, new_h = int(avail_w), int(avail_h)
        if new_w != self.width or new_h != self.height:
            self._resize(new_w, new_h)

        imgui.image(
            imgui.ImTextureRef(self._resolve_tex),
            imgui.ImVec2(avail_w, avail_h),
            uv0=imgui.ImVec2(0, 1),
            uv1=imgui.ImVec2(1, 0),
        )
        self._viewer_size = imgui.get_item_rect_size()
        self.is_scene_hovered = imgui.is_item_hovered()

    # ── Input handling ─────────────────────────────────────────────────
    def handle_input(self):
        """Process mouse input for camera and perturbation. Called by the extension."""
        # Handle space key for play/pause toggle
        if imgui.is_key_pressed(imgui.Key.space):
            if self._extension.playback_state == PlaybackState.PLAYING:
                self._extension._pause()
            else:
                self._extension._play()

        if not self.is_scene_hovered:
            if self.mj_perturb.active != 0:
                self.mj_perturb.active = 0
            return

        import sys
        _IS_MACOS = sys.platform == 'darwin'

        # On macOS: physical Ctrl → Key.left_super/right_super
        # AND macOS converts Ctrl+left-click → right-click at OS level
        if _IS_MACOS:
            ctrl_held = imgui.is_key_down(imgui.Key.left_super) or imgui.is_key_down(imgui.Key.right_super)
            perturb_mouse = imgui.Key.mouse_right
        else:
            ctrl_held = imgui.is_key_down(imgui.Key.left_ctrl) or imgui.is_key_down(imgui.Key.right_ctrl)
            perturb_mouse = imgui.Key.mouse_left
        shift_held = imgui.is_key_down(imgui.Key.left_shift) or imgui.is_key_down(imgui.Key.right_shift)

        # Double-click: select/deselect body
        if imgui.is_mouse_double_clicked(imgui.MouseButton_.left):
            self._pick_body()

        # Ctrl + drag: update perturbation reference (forces applied in physics loop)
        if ctrl_held and self.mj_perturb.select > 0:
            if imgui.is_key_down(perturb_mouse):
                # On drag start: re-init perturb from current body state
                # (mirrors simulate.cc pending_.newperturb logic)
                newperturb = (
                    mujoco.mjtPertBit.mjPERT_ROTATE if shift_held
                    else mujoco.mjtPertBit.mjPERT_TRANSLATE
                )
                if not self.mj_perturb.active:
                    mujoco.mjv_initPerturb(
                        self.model, self.data, self.mj_scene, self.mj_perturb
                    )
                    self.mj_perturb.active = newperturb

                mouse_delta = self._io.mouse_delta
                if shift_held:
                    self._perturb_rotate(mouse_delta)
                else:
                    self._perturb_translate(mouse_delta)
            else:
                self.mj_perturb.active = 0

        elif not ctrl_held:
            self.mj_perturb.active = 0
            # Camera: orbit, pan, zoom
            mouse_interactions(
                self.model, self.mj_scene, self.mj_camera,
                self.width, self.height,
            )

        # Keyboard toggles for visualization flags
        keyboard_interactions(self.mj_scene, self.mj_option)

    def apply_perturbation(self, paused=False):
        """Apply perturbation forces to physics. Call before each mj_step.

        Mirrors MuJoCo simulate.cc Sync():
        1. Zero xfrc_applied (clear stale forces)
        2. applyPerturbPose with flg_paused=0 (mocap only) when running,
           or flg_paused=1 (mocap + dynamic) when paused
        3. applyPerturbForce (writes force from perturb ref → xfrc_applied)
        """
        if self.mj_perturb is None:
            return
        # Clear old perturbation forces
        self.data.xfrc_applied[:] = 0
        if paused:
            mujoco.mjv_applyPerturbPose(self.model, self.data, self.mj_perturb, 1)
        else:
            mujoco.mjv_applyPerturbPose(self.model, self.data, self.mj_perturb, 0)
            mujoco.mjv_applyPerturbForce(self.model, self.data, self.mj_perturb)

    # ── Body selection & perturbation ────────────────────────────────

    def _screen_to_mujoco(self, mouse_pos):
        """Convert ImGui screen coords to MuJoCo framebuffer coords.

        UV is uv0=(0,1), uv1=(1,0): vertical flip only (OpenGL bottom-origin).
        X maps directly; Y is inverted.
        """
        rel_x = mouse_pos.x - self._viewer_pos.x
        rel_y = mouse_pos.y - self._viewer_pos.y
        gl_x = int(rel_x / self._viewer_size.x * self.width)
        gl_y = int((1.0 - rel_y / self._viewer_size.y) * self.height)
        return gl_x, gl_y

    def _pick_body(self):
        """Raycast into scene and select/deselect body under cursor."""
        gl_x, gl_y = self._screen_to_mujoco(self._io.mouse_pos)
        aspect = self.width / self.height

        geom_id = np.array([-1], dtype=np.int32)
        flex_id = np.array([-1], dtype=np.int32)
        skin_id = np.array([-1], dtype=np.int32)
        sel_pos = np.zeros(3, dtype=np.float64)

        body_id = mujoco.mjv_select(
            self.model, self.data, self.mj_option,
            aspect, gl_x / self.width, gl_y / self.height,
            self.mj_scene, sel_pos, geom_id, flex_id, skin_id,
        )

        if body_id > 0:
            self.mj_perturb.select = body_id
            self.mj_perturb.refselpos = sel_pos
            # Use body CoM in local frame so perturbation force has a
            # moment arm relative to the joint (body frame origin).
            body_pos = self.data.xpos[body_id]
            body_mat = self.data.xmat[body_id].reshape(3, 3)
            com_world = self.data.xipos[body_id]
            self.mj_perturb.localpos = body_mat.T @ (com_world - body_pos)
        else:
            self.mj_perturb.select = 0
            self.mj_perturb.active = 0

    def _perturb_translate(self, mouse_delta):
        """Ctrl + drag: update perturbation reference position."""
        dx = mouse_delta.x / self.width
        dy = mouse_delta.y / self.height
        mujoco.mjv_movePerturb(
            self.model, self.data, mujoco.mjtMouse.mjMOUSE_MOVE_H,
            dx, 0.0, self.mj_scene, self.mj_perturb,
        )
        mujoco.mjv_movePerturb(
            self.model, self.data, mujoco.mjtMouse.mjMOUSE_MOVE_V,
            0.0, dy, self.mj_scene, self.mj_perturb,
        )

    def _perturb_rotate(self, mouse_delta):
        """Ctrl + Shift + drag: update perturbation reference orientation."""
        dx = mouse_delta.x / self.width
        dy = mouse_delta.y / self.height
        mujoco.mjv_movePerturb(
            self.model, self.data, mujoco.mjtMouse.mjMOUSE_ROTATE_H,
            dx, 0.0, self.mj_scene, self.mj_perturb,
        )
        mujoco.mjv_movePerturb(
            self.model, self.data, mujoco.mjtMouse.mjMOUSE_ROTATE_V,
            0.0, dy, self.mj_scene, self.mj_perturb,
        )
