""" MuJoCo 3D viewport window """

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
from farms_app.backends.renderer.gl_framebuffer import MSAAFramebuffer
from farms_app.core.window import Window
from farms_app.utils.mujoco import (
    setup_scene, mouse_interactions, keyboard_interactions,
)
from imgui_bundle import imgui

if TYPE_CHECKING:
    from farms_app.extensions.farmsim.extension import FARMSIMExtension


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

        self.fb = None
        self.mj_camera = None
        self.mj_option = None
        self.mj_perturb = None
        self.mj_scene = None
        self.mj_context = None
        self.mj_viewport = None

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
        render_flags = {
            mujoco.mjtRndFlag.mjRND_SKYBOX: True,
            mujoco.mjtRndFlag.mjRND_REFLECTION: True,
            mujoco.mjtRndFlag.mjRND_SHADOW: False,
        }
        (self.mj_camera, self.mj_option, self.mj_perturb,
         self.mj_context, self.mj_scene, self.mj_viewport) = setup_scene(
            self.model, self.width, self.height, render_flags=render_flags,
        )
        self.fb = MSAAFramebuffer(self.width, self.height, samples=4)

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

        if ext.sim is None:
            imgui.text("No simulation loaded")
            return

        self._render_scene()

    def _resize(self, width: int, height: int):
        """Resize framebuffer and MuJoCo viewport to new dimensions."""
        self.width = width
        self.height = height
        self.fb.resize(width, height)
        self.mj_viewport.width = width
        self.mj_viewport.height = height

    def _render_scene(self):
        """Render MuJoCo scene to framebuffer and display as ImGui image."""
        self._viewer_pos = imgui.get_cursor_screen_pos()

        avail_w, avail_h = imgui.get_content_region_avail()
        if avail_w <= 0 or avail_h <= 0:
            return

        new_w, new_h = int(avail_w), int(avail_h)
        if new_w != self.width or new_h != self.height:
            self._resize(new_w, new_h)

        self.fb.bind()
        mujoco.mjv_updateScene(
            self.model, self.data,
            self.mj_option, self.mj_perturb, self.mj_camera,
            mujoco.mjtCatBit.mjCAT_ALL, self.mj_scene,
        )
        mujoco.mjr_render(self.mj_viewport, self.mj_scene, self.mj_context)
        self.fb.unbind()
        self.fb.resolve()

        imgui.image(
            imgui.ImTextureRef(self.fb.texture_id),
            imgui.ImVec2(avail_w, avail_h),
            uv0=imgui.ImVec2(1, 1),
            uv1=imgui.ImVec2(0, 0),
        )
        self._viewer_size = imgui.get_item_rect_size()
        self.is_scene_hovered = imgui.is_item_hovered()

    # ── Input handling ─────────────────────────────────────────────────

    def handle_input(self):
        """Process mouse input for camera and perturbation. Called by the extension."""
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

        Accounts for UV flip (uv0=1,1  uv1=0,0) in imgui.image().
        """
        rel_x = mouse_pos.x - self._viewer_pos.x
        rel_y = mouse_pos.y - self._viewer_pos.y
        gl_x = int((1.0 - rel_x / self._viewer_size.x) * self.width)
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
        # Negate deltas to account for UV flip (uv0=1,1 uv1=0,0)
        dx = -mouse_delta.x / self.width
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
        dx = -mouse_delta.x / self.width
        dy = mouse_delta.y / self.height
        mujoco.mjv_movePerturb(
            self.model, self.data, mujoco.mjtMouse.mjMOUSE_ROTATE_H,
            dx, 0.0, self.mj_scene, self.mj_perturb,
        )
        mujoco.mjv_movePerturb(
            self.model, self.data, mujoco.mjtMouse.mjMOUSE_ROTATE_V,
            0.0, dy, self.mj_scene, self.mj_perturb,
        )
