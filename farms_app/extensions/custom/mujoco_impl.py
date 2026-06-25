""" MuJoCo """
from farms_app.utils.mujoco import setup_scene

import ctypes

import mujoco
import numpy as np
import OpenGL.GL as GL  # type: ignore
from farms_app.backends.renderer.gl_framebuffer import MSAAFramebuffer
from farms_core import pylog
from imgui_bundle import imgui

from farms_app.core.extension import Extension
from imgui_bundle import portable_file_dialogs as pfd
from farms_app.core.window import Window


MJ_IMGUI_KEYMAP = {
    # special keys
    "/": imgui.Key.slash,
    "\\": imgui.Key.backslash,
    ",": imgui.Key.comma,
    ".": imgui.Key.period,
    ";": imgui.Key.semicolon,
    "'": imgui.Key.apostrophe,
    "[": imgui.Key.left_bracket,
    "]": imgui.Key.right_bracket,
    "-": imgui.Key.minus,
    "=": imgui.Key.equal,
    "`": imgui.Key.grave_accent,
    # letters A–Z
    **{chr(c): getattr(imgui.Key, chr(c).lower()) for c in range(ord("A"), ord("Z")+1)},
    **{str(i): getattr(imgui.Key, f"_{i}") for i in range(6)},
}


_mjGEOMSTRING = (
    (
        "Geom1", "1", "0",
        "Geom2", "1", "1",
        "Geom3", "1", "2",
        "Geom4", "0", "3",
        "Geom5", "0", "4",
        "Geom6", "0", "5",
    ),
)


class MuJoCoWindow(Window):
    """ MuJoCo Window """

    def __init__(self, extension, window_size = (1280, 720)):
        name: str = "MuJoCo"
        super().__init__(name, extension)
        self._io = imgui.get_io()
        self.model = None
        self.data = None
        self._io = imgui.get_io()
        self.width = window_size[0]
        self.height = window_size[1]

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

    def on_initialize(self):
        """ Initialize """
        pass

    def on_render(self):
        """ Render main extension dockspace """
        if self.data is not None and self.model is not None:
            self.run_simulation()
            imgui.text(f"select: {self.mj_perturb.select}  active: {self.mj_perturb.active}")
            imgui.text(f"xfrc[select]: {self.data.xfrc_applied[max(self.mj_perturb.select, 0)]}")
        else:
            if imgui.button("Load mjcf"):
                self.result = pfd.open_file("Load MJCF", default_path="", filters=("*.xml",),).result()
                if self.result:
                    self.model = mujoco.MjModel.from_xml_path(self.result[0])
                    self.data = mujoco.MjData(self.model)
                    # Setup mujoco scene
                    render_flags = {
                        mujoco.mjtRndFlag.mjRND_SKYBOX: True,
                        mujoco.mjtRndFlag.mjRND_REFLECTION: True,
                        mujoco.mjtRndFlag.mjRND_SHADOW: True,
                    }
                    (self.mj_camera, self.mj_option, self.mj_perturb,
                     self.mj_context, self.mj_scene, self.mj_viewport) = setup_scene(
                         self.model, self.width, self.height, render_flags=render_flags,
                    )
                    self.fb = MSAAFramebuffer(self.width, self.height, samples=4)

    def run_simulation(self):
        """ Run Simulation """
        self.start_time = self.data.time
        while (self.data.time - self.start_time < 1.0/60.0):
            self.apply_perturbation()
            mujoco.mj_step(self.model, self.data)

        self._viewer_pos = imgui.get_cursor_screen_pos()

        self.fb.bind()
        mujoco.mjv_updateScene(
            self.model, self.data,
            self.mj_option, self.mj_perturb, self.mj_camera,
            mujoco.mjtCatBit.mjCAT_ALL, self.mj_scene,
        )
        mujoco.mjr_render(self.mj_viewport, self.mj_scene, self.mj_context)
        self.fb.unbind()
        self.fb.resolve()

        # Fit image to available space preserving aspect ratio
        avail_w, avail_h = imgui.get_content_region_avail()
        if avail_w <= 0 or avail_h <= 0:
            return

        aspect = self.width / self.height
        if avail_w / avail_h > aspect:
            draw_h = avail_h
            draw_w = aspect * draw_h
        else:
            draw_w = avail_w
            draw_h = draw_w / aspect

        imgui.image(
            imgui.ImTextureRef(self.fb.texture_id),
            imgui.ImVec2(draw_w, draw_h),
            uv0=imgui.ImVec2(1, 1),
            uv1=imgui.ImVec2(0, 0),
        )
        self._viewer_size = imgui.get_item_rect_size()
        self.is_scene_hovered = imgui.is_item_hovered()

    def handle_input(self):
        if not self.is_scene_hovered:
            return

        # Keyboard
        self.keyboard_interactions()
        # Mouse
        self.mouse_interactions()

    def __mj_keys(self, mjSTRING: tuple[str, str, str], mj_flags):
        for j, _opt in enumerate(mjSTRING):
            key_str = _opt[2]
            if not key_str:
                continue        # Skip if no key assigned
            key_enum = MJ_IMGUI_KEYMAP.get(key_str)
            if key_enum is None:
                continue
            if imgui.is_key_pressed(key_enum):
                mj_flags[j] = not mj_flags[j]

    def keyboard_interactions(self):
        """ keyboard interactions """
        self.__mj_keys(mujoco.mjRNDSTRING, self.mj_scene.flags)
        self.__mj_keys(mujoco.mjVISSTRING, self.mj_option.flags)
        self.__mj_keys(_mjGEOMSTRING, self.mj_option.geomgroup)

    def _screen_to_viewport(self, mouse_pos):
        """Convert ImGui screen coords to MuJoCo framebuffer coords.

        Accounts for UV flip (uv0=1,1  uv1=0,0) in imgui.image().
        """
        rel_x = mouse_pos.x - self._viewer_pos.x
        rel_y = mouse_pos.y - self._viewer_pos.y
        gl_x = int((1.0 - rel_x / self._viewer_size.x) * self.width)
        gl_y = int((1.0 - rel_y / self._viewer_size.y) * self.height)
        return gl_x, gl_y

    def _pick_body(self):
        """Double-click: select body under cursor."""
        gl_x, gl_y = self._screen_to_viewport(self._io.mouse_pos)
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
            # Compute click point in body-local frame (mirrors simulate.cc)
            body_pos = self.data.xpos[body_id]
            body_mat = self.data.xmat[body_id].reshape(3, 3)
            self.mj_perturb.localpos = body_mat.T @ (sel_pos - body_pos)
        else:
            self.mj_perturb.select = 0
            self.mj_perturb.active = 0

    def apply_perturbation(self):
        """Apply perturbation forces. Call before mj_step.

        Mirrors simulate.cc Sync():
        1. Zero xfrc_applied
        2. applyPerturbPose (mocap only when running)
        3. applyPerturbForce
        """
        self.data.xfrc_applied[:] = 0
        mujoco.mjv_applyPerturbPose(self.model, self.data, self.mj_perturb, 0)
        mujoco.mjv_applyPerturbForce(self.model, self.data, self.mj_perturb)

    def mouse_interactions(self):
        """ Mouse interactions """
        mouse_delta = self._io.mouse_delta
        mouse_wheel = self._io.mouse_wheel
        import sys
        _IS_MACOS = sys.platform == 'darwin'

        # On macOS: physical Ctrl → Key.left_super/right_super
        # AND macOS converts Ctrl+left-click → right-click at OS level
        if _IS_MACOS:
            ctrl_held = imgui.is_key_down(imgui.Key.left_super) or imgui.is_key_down(imgui.Key.right_super)
            perturb_mouse = imgui.Key.mouse_right   # Ctrl+click arrives as right-click
        else:
            ctrl_held = imgui.is_key_down(imgui.Key.left_ctrl) or imgui.is_key_down(imgui.Key.right_ctrl)
            perturb_mouse = imgui.Key.mouse_left
        shift_held = imgui.is_key_down(imgui.Key.left_shift) or imgui.is_key_down(imgui.Key.right_shift)

        # Double-click: select/deselect body
        if imgui.is_mouse_double_clicked(imgui.MouseButton_.left):
            self._pick_body()

        # Ctrl + drag: perturbation (macOS: Ctrl+click = right-click)
        if ctrl_held and self.mj_perturb.select > 0:
            if imgui.is_key_down(perturb_mouse):
                newperturb = (
                    mujoco.mjtPertBit.mjPERT_ROTATE if shift_held
                    else mujoco.mjtPertBit.mjPERT_TRANSLATE
                )
                if not self.mj_perturb.active:
                    mujoco.mjv_initPerturb(
                        self.model, self.data, self.mj_scene, self.mj_perturb
                    )
                    self.mj_perturb.active = newperturb

                # Negate deltas to account for UV flip (uv0=1,1 uv1=0,0)
                dx = -mouse_delta.x / self.width
                dy = mouse_delta.y / self.height
                if shift_held:
                    mujoco.mjv_movePerturb(
                        self.model, self.data, mujoco.mjtMouse.mjMOUSE_ROTATE_H,
                        dx, 0.0, self.mj_scene, self.mj_perturb,
                    )
                    mujoco.mjv_movePerturb(
                        self.model, self.data, mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                        0.0, dy, self.mj_scene, self.mj_perturb,
                    )
                else:
                    mujoco.mjv_movePerturb(
                        self.model, self.data, mujoco.mjtMouse.mjMOUSE_MOVE_H,
                        dx, 0.0, self.mj_scene, self.mj_perturb,
                    )
                    mujoco.mjv_movePerturb(
                        self.model, self.data, mujoco.mjtMouse.mjMOUSE_MOVE_V,
                        0.0, dy, self.mj_scene, self.mj_perturb,
                    )
            else:
                self.mj_perturb.active = 0

        elif not ctrl_held:
            self.mj_perturb.active = 0
            # Camera: left-drag = orbit, right-drag = pan, wheel = zoom
            if imgui.is_key_down(imgui.Key.mouse_left):
                mujoco.mjv_moveCamera(
                    self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_H,
                    -mouse_delta.x / self.width, 0.0,
                    self.mj_scene, self.mj_camera,
                )
                mujoco.mjv_moveCamera(
                    self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                    0.0, mouse_delta.y / self.height,
                    self.mj_scene, self.mj_camera,
                )
            elif imgui.is_key_down(imgui.Key.mouse_right):
                mujoco.mjv_moveCamera(
                    self.model, mujoco.mjtMouse.mjMOUSE_MOVE_H,
                    -mouse_delta.x / self.width, 0.0,
                    self.mj_scene, self.mj_camera,
                )
                mujoco.mjv_moveCamera(
                    self.model, mujoco.mjtMouse.mjMOUSE_MOVE_V,
                    0.0, mouse_delta.y / self.height,
                    self.mj_scene, self.mj_camera,
                )
            elif imgui.is_key_down(imgui.Key.mouse_wheel_y):
                mujoco.mjv_moveCamera(
                    self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                    0.0, np.sign(mouse_wheel) * 0.05,
                    self.mj_scene, self.mj_camera,
                )


class MuJoCoExtension(Extension):
    """ MuJoCo """

    def __init__(self):
        super().__init__(name="MuJoCo")

        # Register windows
        self._mujoco_win: MuJoCoWindow = MuJoCoWindow(self)
        self.register_window(self._mujoco_win)
        self._mujoco_win.initialize()

    def __del__(self):
        print("Terminating MuJoCo Extension")

    def get_name(self) -> str:
        return "MuJoCo"

    def cleanup(self):
        pass

    def get_dependencies(self):
        pass

    def on_event(self):
        if self._mujoco_win.model and self._mujoco_win.data:
            self._mujoco_win.handle_input()
