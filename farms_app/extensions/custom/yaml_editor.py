from farms_app.plugins.base import CustomWidget

from farms_core.io.yaml import read_yaml
import yaml
from imgui_bundle import imgui


yaml_data = read_yaml("/Users/tatarama/data/mouse/share/quadruped-jon/animat_options.yaml")


class YAMLEditorPlugin(CustomWidget):
    """ Yaml Editor Plugin """

    def __init__(self):
        self.show_window = True

        self.focused_path = []  # Track current focus path like ["simulation", "timestep"]
        self.focused_yaml_text = ""  # Text for focused section only

    def get_name(self) -> str:
        return "YAML Editor"

    def render(self) -> None:
        if not self.show_window:
            return

        expanded, self.show_window = imgui.begin(
            "YAML Editor", self.show_window,
            flags=(imgui.WindowFlags_.horizontal_scrollbar),
        )
        # yaml_editor(yaml_data)
        # Left side - YAML Editor (60% width)
        window_width = imgui.get_window_width()
        editor_width = window_width * 0.6

        if imgui.begin_child("yaml_editor", imgui.ImVec2(editor_width, 0), True):
            imgui.text("YAML Editor")
            imgui.separator()
            self.yaml_editor(yaml_data)

        imgui.end_child()

        imgui.same_line()

        # Right side - Parsed Output (40% width)
        if imgui.begin_child("yaml_output", imgui.ImVec2(0, 0), True):
            imgui.text("Parsed YAML")
            imgui.separator()

            # Display parsed content as text
            # Show focused path
            if self.focused_path:
                path_str = " → ".join(self.focused_path)
                imgui.text_colored(imgui.ImVec4(0.7, 0.7, 1.0, 1.0), f"Focus: {path_str}")
                imgui.separator()

            # Text editor - show full or focused content
            if self.focused_yaml_text:
                display_text = self.focused_yaml_text

                changed, new_text = imgui.input_text_multiline(
                    "##yaml_input",
                    display_text,
                    imgui.ImVec2(-1, -1),
                    imgui.InputTextFlags_.allow_tab_input | imgui.InputTextFlags_.read_only
                )

            # Note: Made read-only for focused sections to avoid conflicts
            # Full editing still works in tree view

        imgui.end_child()
        imgui.end()



    def yaml_editor(self, data: dict, indent=0, path=[]):
        """ Enhanced YAML Editor with proper indentation """

        for name_d1, values_d1 in data.items():
            current_path = path + [name_d1]
            # Create indentation
            imgui.indent(indent * 20.0)  # 20 pixels per indent level

            if isinstance(values_d1, dict):
                # Nested dictionary - use collapsing header
                if imgui.collapsing_header(f"{name_d1}"):
                    self._set_focus(current_path, values_d1)
                    self.yaml_editor(values_d1, indent + 1, current_path)

            elif isinstance(values_d1, list):
                # Check if it's a list of numbers - display as inline editable fields
                if len(values_d1) > 0 and all(isinstance(x, (int, float)) for x in values_d1):
                    imgui.text(f"{name_d1} [{len(values_d1)}]:")

                    # Set focus when any input is active
                    if self._is_any_input_active(f"list_num_{name_d1}"):
                        self._set_focus(current_path, values_d1)

                    # Display each number as inline input field
                    for i, item in enumerate(values_d1):
                        if i > 0:
                            imgui.same_line()

                        imgui.push_item_width(80)  # Fixed width for each input

                        if isinstance(item, int):
                            changed, new_value = imgui.input_int(f"##list_num_{name_d1}_{i}_{indent}", item)
                        else:
                            changed, new_value = imgui.input_float(f"##list_num_{name_d1}_{i}_{indent}", item)

                        imgui.pop_item_width()

                        if changed:
                            data[name_d1][i] = new_value
                else:
                    # Regular list - show count and expandable items
                    if imgui.collapsing_header(f"{name_d1} [{len(values_d1)} items]"):
                        for i, item in enumerate(values_d1):
                            imgui.indent((indent + 1) * 20.0)

                            if isinstance(item, dict):
                                if imgui.collapsing_header(f"[{i}]"):
                                    self.yaml_editor(item, indent + 2)
                            elif isinstance(item, str):
                                # Editable list item (string)
                                imgui.text(f"[{i}]:")
                                imgui.same_line()
                                changed, new_value = imgui.input_text(f"##list_str_{name_d1}_{i}_{indent}", item)
                                if changed:
                                    data[name_d1][i] = new_value
                            elif isinstance(item, (int, float)):
                                # Editable list item (numeric)
                                imgui.text(f"[{i}]:")
                                imgui.same_line()
                                if isinstance(item, int):
                                    changed, new_value = imgui.input_int(f"##list_int_{name_d1}_{i}_{indent}", item)
                                else:
                                    changed, new_value = imgui.input_float(f"##list_float_{name_d1}_{i}_{indent}", item)
                                if changed:
                                    data[name_d1][i] = new_value
                            elif isinstance(item, bool):
                                # Editable list item (boolean)
                                changed, new_value = imgui.checkbox(f"[{i}] {name_d1}", item)
                                if changed:
                                    data[name_d1][i] = new_value
                            else:
                                imgui.text(f"[{i}]: {item}")

                            imgui.unindent((indent + 1) * 20.0)

            elif isinstance(values_d1, str):
                # Editable string value
                imgui.text(f"{name_d1}:")
                imgui.same_line()
                changed, new_value = imgui.input_text(f"##str_{name_d1}_{indent}", values_d1)
                # Check if this input is active/focused
                if imgui.is_item_active():
                    self._set_focus(current_path, values_d1)
                if changed:
                    data[name_d1] = new_value

            elif isinstance(values_d1, (int, float)):
                # Editable numeric value
                imgui.text(f"{name_d1}:")
                imgui.same_line()
                if isinstance(values_d1, int):
                    changed, new_value = imgui.input_int(f"##int_{name_d1}_{indent}", values_d1)
                else:
                    changed, new_value = imgui.input_float(f"##float_{name_d1}_{indent}", values_d1)
                if imgui.is_item_active():
                    self._set_focus(current_path, values_d1)
                if changed:
                    data[name_d1] = new_value

            elif isinstance(values_d1, bool):
                # Editable boolean value
                changed, new_value = imgui.checkbox(f"{name_d1}", values_d1)
                if imgui.is_item_active():
                     self._set_focus(current_path, values_d1)
                if changed:
                    data[name_d1] = new_value

            elif values_d1 is None:
                # Editable null value (as string input)
                imgui.text(f"{name_d1}: null")
                imgui.same_line()
                if imgui.small_button(f"Edit##null_{name_d1}_{indent}"):
                    data[name_d1] = ""  # Convert to empty string for editing

            else:
                # Unknown type
                imgui.text(f"{name_d1}: {str(values_d1)}")

            # Remove indentation for next item
            imgui.unindent(indent * 20.0)

    def _set_focus(self, path, value):
        """Set the currently focused path and generate focused YAML text"""
        self.focused_path = path.copy()
        self._generate_focused_yaml(value)

    def _generate_focused_yaml(self, focused_data):
        """Generate YAML text for just the focused section"""
        try:
            if isinstance(focused_data, dict):
                # Create a temporary dict with just the focused section
                temp_dict = {self.focused_path[-1]: focused_data}
                self.focused_yaml_text = yaml.dump(temp_dict, default_flow_style=False, indent=2)
            else:
                # For leaf values, show the key-value pair
                key = self.focused_path[-1] if self.focused_path else "value"
                temp_dict = {key: focused_data}
                self.focused_yaml_text = yaml.dump(temp_dict, default_flow_style=False, indent=2)
        except:
            self.focused_yaml_text = "# Focus on a section to see YAML"

    def _is_any_input_active(self, base_id):
        """Check if any input with the base ID is currently active"""
        # This is a helper to detect if any numeric list input is focused
        # ImGui doesn't have a direct way to check this, so we'll track it differently
        return False  # Simplified for now
