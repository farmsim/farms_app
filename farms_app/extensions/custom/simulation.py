from farms_app.plugins.base import BasePlugin
from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd


farms_info = {
    "name": "Simulation Plugin",
    "stage": "simulation",
    "version": (1, 0, 0),
    "author": "farms"
}


class ParameterEditorPlugin(BasePlugin):
    """Simple plugin to edit simulation parameters"""

    def __init__(self):
        self.show_window = True
        self.window_name = "Simulation"
        self.gravity = -9.81
        self.timestep = 0.001
        self.simulation_time = 10.0
        self.physics_engine = 0  # 0=MuJoCo, 1=Bullet

    def get_name(self) -> str:
        return "Parameter Editor"

    def render(self) -> None:

        imgui.text("Physics Settings")

        changed, self.gravity = imgui.slider_float("Gravity", self.gravity, -20.0, 0.0)
        changed, self.timestep = imgui.slider_float("Timestep", self.timestep, 0.0001, 0.01, "%.4f")
        changed, self.simulation_time = imgui.slider_float("Sim Time", self.simulation_time, 1.0, 60.0)

        physics_engines = ["MuJoCo", "Bullet"]
        changed, self.physics_engine = imgui.combo("Physics Engine", self.physics_engine, physics_engines)

        if imgui.button("Apply Settings"):
            print(f"Applied: gravity={self.gravity}, timestep={self.timestep}")

        imgui.separator_text("File dialogs")
        if imgui.button("Simulation Options"):
            self.open_file_dialog = pfd.open_file("Simulation Options")
            self.simulation_options = self.open_file_dialog.result()
        if imgui.button("Animat Options"):
            self.open_file_dialog = pfd.open_file("Animat Options")
            self.animat_options = self.open_file_dialog.result()


def register():
    """ Registration """
    print("Register plugin")


def unregister():
    """ Unregesiter """
    print("Unregistering plugin")
