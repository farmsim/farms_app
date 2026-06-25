"""Generic YAML/XML config editor window.

A reusable Window that any extension can register. Loads YAML or XML
files, renders an editable tree, and saves back.

Usage:
    editor = ConfigEditorWindow(extension, name="My Config")
    extension.register_window(editor)
    editor.load_file("path/to/config.yaml")
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from farms_core.io.yaml import read_yaml
from farms_app.core.window import Window
from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd
from imgui_bundle.immapp import icons_fontawesome_6 as fa6

if TYPE_CHECKING:
    from farms_app.core.extension import Extension


class ConfigEditorWindow(Window):
    """Editable tree editor for YAML and XML config files."""

    def __init__(self, extension: Extension, name: str = "Config Editor"):
        super().__init__(name, extension)
        self._file_path = None
        self._file_type = None  # "yaml" or "xml"
        self._data = None       # dict for YAML, ElementTree for XML
        self._dirty = False
        self._search_text = ""

    @property
    def file_path(self):
        return self._file_path

    def load_file(self, path: str):
        """Load a YAML or XML file for editing."""
        path = str(path)
        suffix = Path(path).suffix.lower()

        if suffix in (".yaml", ".yml"):
            self._data = read_yaml(path)
            self._file_type = "yaml"
        elif suffix in (".xml", ".sdf", ".urdf", ".mjcf"):
            self._data = ET.parse(path)
            self._file_type = "xml"
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        self._file_path = path
        self._dirty = False

    def save(self, path: str = None):
        """Save the current data back to file."""
        path = path or self._file_path
        if path is None or self._data is None:
            return

        if self._file_type == "yaml":
            with open(path, "w") as f:
                yaml.dump(self._data, f, default_flow_style=False, indent=2)
        elif self._file_type == "xml":
            self._data.write(path, xml_declaration=True)

        self._file_path = path
        self._dirty = False

    def on_render(self):
        self._render_toolbar()
        imgui.separator()

        if self._data is None:
            imgui.text("No file loaded")
            return

        # Scrollable tree
        flags = imgui.WindowFlags_.horizontal_scrollbar
        if imgui.begin_child("##tree", imgui.ImVec2(0, 0), imgui.ChildFlags_.none, flags):
            if self._file_type == "yaml":
                self._render_yaml(self._data, "root")
            elif self._file_type == "xml":
                self._render_xml(self._data.getroot())
        imgui.end_child()

    # Toolbar
    def _render_toolbar(self):
        if imgui.button(f"{fa6.ICON_FA_FOLDER_OPEN}  Open"):
            result = pfd.open_file(
                "Open config", "", ["*.yaml *.yml *.xml *.sdf *.urdf *.mjcf"],
            ).result()
            if result:
                self.load_file(result[0])

        imgui.same_line()
        has_file = self._data is not None
        if not has_file:
            imgui.begin_disabled()

        if imgui.button(f"{fa6.ICON_FA_FLOPPY_DISK}  Save"):
            self.save()

        imgui.same_line()
        if imgui.button(f"{fa6.ICON_FA_FLOPPY_DISK}  Save As"):
            result = pfd.save_file(
                "Save config", self._file_path or "",
                ["*.yaml *.yml *.xml *.sdf"],
            ).result()
            if result:
                self.save(result)

        if not has_file:
            imgui.end_disabled()

        # File info
        if self._file_path:
            imgui.same_line()
            label = Path(self._file_path).name
            if self._dirty:
                label += " *"
            imgui.text_disabled(label)

    # YAML rendering
    def _render_yaml(self, data: Any, key: str, parent=None, parent_key=None):
        """Recursively render a YAML value as an editable tree."""
        if isinstance(data, dict):
            flags = imgui.TreeNodeFlags_.default_open if parent is None else 0
            if imgui.tree_node_ex(f"{key}  ({len(data)})", flags):
                for k in list(data.keys()):
                    self._render_yaml(data[k], str(k), parent=data, parent_key=k)
                imgui.tree_pop()

        elif isinstance(data, list):
            if len(data) == 0:
                imgui.text(f"{key}: []")
            elif all(isinstance(x, (int, float)) for x in data) and len(data) <= 8:
                # Compact numeric list — inline editors
                self._render_numeric_list(data, key, parent, parent_key)
            elif all(isinstance(x, (dict, list)) for x in data):
                # List of complex objects
                if imgui.tree_node(f"{key}  [{len(data)}]"):
                    for i, item in enumerate(data):
                        self._render_yaml(item, f"[{i}]", parent=data, parent_key=i)
                    imgui.tree_pop()
            else:
                # Mixed or simple list
                if imgui.tree_node(f"{key}  [{len(data)}]"):
                    for i, item in enumerate(data):
                        self._render_yaml(item, f"[{i}]", parent=data, parent_key=i)
                    imgui.tree_pop()

        else:
            # Leaf value — editable
            self._render_leaf(data, key, parent, parent_key)

    def _render_leaf(self, value, key: str, parent, parent_key):
        """Render an editable leaf value."""
        if parent is None:
            imgui.text(f"{key}: {value}")
            return

        uid = f"##{id(parent)}_{parent_key}"
        imgui.text_disabled(f"{key}:")
        imgui.same_line()

        if isinstance(value, bool):
            changed, new_val = imgui.checkbox(uid, value)
            if changed:
                parent[parent_key] = new_val
                self._dirty = True

        elif isinstance(value, int):
            imgui.set_next_item_width(120)
            changed, new_val = imgui.input_int(uid, value)
            if changed:
                parent[parent_key] = new_val
                self._dirty = True

        elif isinstance(value, float):
            imgui.set_next_item_width(120)
            changed, new_val = imgui.input_double(uid, value, 0.0, 0.0, "%.6g")
            if changed:
                parent[parent_key] = new_val
                self._dirty = True

        elif isinstance(value, str):
            imgui.set_next_item_width(max(200, imgui.calc_text_size(value).x + 30))
            changed, new_val = imgui.input_text(uid, value)
            if changed:
                parent[parent_key] = new_val
                self._dirty = True

        elif value is None:
            imgui.text_disabled("null")
            imgui.same_line()
            if imgui.small_button(f"Edit{uid}"):
                parent[parent_key] = ""
                self._dirty = True

        else:
            imgui.text(str(value))

    def _render_numeric_list(self, data: list, key: str, parent, parent_key):
        """Render a short numeric list as inline editors."""
        imgui.text_disabled(f"{key}:")
        imgui.same_line()
        for i, item in enumerate(data):
            if i > 0:
                imgui.same_line()
            imgui.push_item_width(80)
            uid = f"##{id(data)}_{i}"
            if isinstance(item, int):
                changed, new_val = imgui.input_int(uid, item)
            else:
                changed, new_val = imgui.input_double(uid, float(item), 0.0, 0.0, "%.4g")
            imgui.pop_item_width()
            if changed:
                data[i] = new_val
                self._dirty = True

    # XML rendering
    def _render_xml(self, element: ET.Element, depth: int = 0):
        """Recursively render an XML element as an editable tree."""
        tag = element.tag
        has_children = len(element) > 0
        has_attribs = len(element.attrib) > 0
        has_text = element.text and element.text.strip()

        # Build display label
        label = f"<{tag}>"
        if 'name' in element.attrib:
            label = f"<{tag} name=\"{element.attrib['name']}\">"

        flags = imgui.TreeNodeFlags_.default_open if depth < 1 else 0

        if has_children or has_attribs:
            if imgui.tree_node_ex(f"{label}##{id(element)}", flags):
                # Attributes
                if has_attribs:
                    for attr_name in list(element.attrib.keys()):
                        uid = f"##{id(element)}_{attr_name}"
                        imgui.text_disabled(f"@{attr_name}:")
                        imgui.same_line()
                        imgui.set_next_item_width(
                            max(150, imgui.calc_text_size(element.attrib[attr_name]).x + 30)
                        )
                        changed, new_val = imgui.input_text(uid, element.attrib[attr_name])
                        if changed:
                            element.attrib[attr_name] = new_val
                            self._dirty = True

                # Text content
                if has_text:
                    self._render_xml_text(element)

                # Children
                for child in element:
                    self._render_xml(child, depth + 1)

                imgui.tree_pop()
        else:
            # Leaf element
            if has_text:
                imgui.text_disabled(f"<{tag}>:")
                imgui.same_line()
                self._render_xml_text(element)
            else:
                imgui.text(f"<{tag}/>")

    def _render_xml_text(self, element: ET.Element):
        """Render editable text content of an XML element."""
        text = element.text.strip() if element.text else ""
        uid = f"##text_{id(element)}"

        # Try to parse as number for appropriate editor
        try:
            val = float(text)
            imgui.set_next_item_width(120)
            changed, new_val = imgui.input_double(uid, val, 0.0, 0.0, "%.6g")
            if changed:
                element.text = f"{new_val:.6g}"
                self._dirty = True
            return
        except (ValueError, TypeError):
            pass

        imgui.set_next_item_width(max(150, imgui.calc_text_size(text).x + 30))
        changed, new_val = imgui.input_text(uid, text)
        if changed:
            element.text = new_val
            self._dirty = True
