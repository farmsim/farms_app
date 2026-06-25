""" MuJoCo utilities: scene setup, camera controls, keyboard/mouse interaction """

import mujoco
import numpy as np
from imgui_bundle import imgui


# ImGui key mapping for MuJoCo toggle keys
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

# Geometry group toggle strings (matches MuJoCo's mjRNDSTRING/mjVISSTRING format)
MJ_GEOMSTRING = (
    (
        "Geom1", "1", "0",
        "Geom2", "1", "1",
        "Geom3", "1", "2",
        "Geom4", "0", "3",
        "Geom5", "0", "4",
        "Geom6", "0", "5",
    ),
)


def setup_scene(model, width, height, render_flags=None):
    """Create and configure MuJoCo visualization objects.

    Args:
        model: MjModel instance
        width: Viewport width
        height: Viewport height
        render_flags: Optional dict of {mjtRndFlag: bool} to set on the scene

    Returns:
        Tuple of (camera, option, perturb, context, scene, viewport)
    """
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = True
    model.vis.headlight.ambient[:] = [0.6] * 3
    model.vis.headlight.diffuse[:] = [0.4] * 3
    model.vis.headlight.specular[:] = [0.5] * 3
    perturb = mujoco.MjvPerturb()
    mujoco.mjv_defaultCamera(camera)
    mujoco.mjv_defaultPerturb(perturb)
    mujoco.mjv_defaultOption(option)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
    scene = mujoco.MjvScene(model, maxgeom=1000000)
    viewport = mujoco.MjrRect(0, 0, width, height)

    if render_flags:
        for flag, value in render_flags.items():
            scene.flags[flag] = value

    return camera, option, perturb, context, scene, viewport


def update_and_render(model, data, option, camera, scene, viewport, context):
    """Update MuJoCo scene and render to the current framebuffer.

    Call between framebuffer.bind() and framebuffer.unbind().
    """
    mujoco.mjv_updateScene(
        model, data, option, None, camera,
        mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    mujoco.mjr_render(viewport, scene, context)


def toggle_keys(mj_string, mj_flags):
    """Process keyboard toggles for MuJoCo visualization flags.

    Args:
        mj_string: MuJoCo string tuple (e.g., mjRNDSTRING, mjVISSTRING, MJ_GEOMSTRING)
        mj_flags: The flags array to toggle
    """
    for j, _opt in enumerate(mj_string):
        key_str = _opt[2]
        if not key_str:
            continue
        key_enum = MJ_IMGUI_KEYMAP.get(key_str)
        if key_enum is None:
            continue
        if imgui.is_key_pressed(key_enum):
            mj_flags[j] = not mj_flags[j]


def keyboard_interactions(scene, option):
    """Handle keyboard toggles for render, visualization, and geom flags."""
    toggle_keys(mujoco.mjRNDSTRING, scene.flags)
    toggle_keys(mujoco.mjVISSTRING, option.flags)
    toggle_keys(MJ_GEOMSTRING, option.geomgroup)


def mouse_interactions(model, scene, camera, width, height):
    """Handle mouse camera controls: orbit, pan, zoom.

    Args:
        model: MjModel
        scene: MjvScene
        camera: MjvCamera
        width: Viewport width (for normalizing mouse delta)
        height: Viewport height (for normalizing mouse delta)
    """
    io = imgui.get_io()
    mouse_delta = io.mouse_delta
    mouse_wheel = io.mouse_wheel

    if imgui.is_key_down(imgui.Key.mouse_left):
        mujoco.mjv_moveCamera(
            model, mujoco.mjtMouse.mjMOUSE_ROTATE_H,
            -mouse_delta.x / width, 0.0, scene, camera,
        )
        mujoco.mjv_moveCamera(
            model, mujoco.mjtMouse.mjMOUSE_ROTATE_V,
            0.0, mouse_delta.y / height, scene, camera,
        )
    elif imgui.is_key_down(imgui.Key.mouse_right):
        mujoco.mjv_moveCamera(
            model, mujoco.mjtMouse.mjMOUSE_MOVE_H,
            -mouse_delta.x / width, 0.0, scene, camera,
        )
        mujoco.mjv_moveCamera(
            model, mujoco.mjtMouse.mjMOUSE_MOVE_V,
            0.0, mouse_delta.y / height, scene, camera,
        )
    elif imgui.is_key_down(imgui.Key.mouse_wheel_y):
        mujoco.mjv_moveCamera(
            model, mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0, np.sign(mouse_wheel) * 0.05, scene, camera,
        )
