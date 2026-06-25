"""Lightweight frame timer for per-extension and per-window profiling.

Instruments ExtensionManager.tick() to measure time spent in each
extension phase (update, event, render) and each window's render call.
Renders an overlay with a sortable breakdown.
"""

import time
from collections import deque
from dataclasses import dataclass, field

from imgui_bundle import imgui, em_to_vec2


@dataclass
class TimingSample:
    """One frame's timing for a named scope."""
    update_ms: float = 0.0
    event_ms: float = 0.0
    render_ms: float = 0.0
    windows: dict[str, float] = field(default_factory=dict)  # window_name → ms

    @property
    def total_ms(self):
        return self.update_ms + self.event_ms + self.render_ms


class FrameTimer:
    """Collects per-extension timing and renders an overlay.

    Usage:
        timer = FrameTimer()

        # In tick(), wrap each phase:
        timer.begin_frame()
        for ext in extensions:
            timer.begin_scope(ext.name)
            timer.begin_phase("update")
            ext.on_update(dt)
            timer.end_phase("update")
            ...
            timer.end_scope()
        timer.end_frame()

        # Render overlay:
        timer.render_overlay()
    """

    def __init__(self, history_size: int = 120):
        self.enabled = False
        self._history_size = history_size
        self._history: dict[str, deque[TimingSample]] = {}
        self._frame_total = deque(maxlen=history_size)

        # Current frame state
        self._current_scope = None
        self._current_sample = None
        self._phase_start = 0.0
        self._frame_start = 0.0

        # Window timing
        self._window_start = 0.0

    def begin_frame(self):
        if not self.enabled:
            return
        self._frame_start = time.perf_counter()

    def end_frame(self):
        if not self.enabled:
            return
        elapsed = (time.perf_counter() - self._frame_start) * 1000.0
        self._frame_total.append(elapsed)

    def begin_scope(self, name: str):
        if not self.enabled:
            return
        if name not in self._history:
            self._history[name] = deque(maxlen=self._history_size)
        self._current_scope = name
        self._current_sample = TimingSample()

    def end_scope(self):
        if not self.enabled or self._current_sample is None:
            return
        self._history[self._current_scope].append(self._current_sample)
        self._current_scope = None
        self._current_sample = None

    def begin_phase(self, phase: str):
        if not self.enabled:
            return
        self._phase_start = time.perf_counter()

    def end_phase(self, phase: str):
        if not self.enabled or self._current_sample is None:
            return
        elapsed = (time.perf_counter() - self._phase_start) * 1000.0
        if phase == "update":
            self._current_sample.update_ms = elapsed
        elif phase == "event":
            self._current_sample.event_ms = elapsed
        elif phase == "render":
            self._current_sample.render_ms = elapsed

    def begin_window(self, window_name: str):
        if not self.enabled:
            return
        self._window_start = time.perf_counter()

    def end_window(self, window_name: str):
        if not self.enabled or self._current_sample is None:
            return
        elapsed = (time.perf_counter() - self._window_start) * 1000.0
        self._current_sample.windows[window_name] = elapsed

    def render_overlay(self):
        """Render the frame timer overlay window."""
        if not self.enabled:
            return

        imgui.set_next_window_size(em_to_vec2(24, 19), imgui.Cond_.first_use_ever)
        imgui.set_next_window_bg_alpha(0.85)
        expanded, self.enabled = imgui.begin("Frame Timer", self.enabled)
        if not expanded:
            imgui.end()
            return

        # Frame total
        if self._frame_total:
            avg_frame = sum(self._frame_total) / len(self._frame_total)
            fps = 1000.0 / avg_frame if avg_frame > 0 else 0
            imgui.text(f"Frame: {avg_frame:.1f} ms  ({fps:.0f} fps)")
        imgui.separator()

        # Per-extension table
        flags = (
            imgui.TableFlags_.borders_inner_h |
            imgui.TableFlags_.row_bg |
            imgui.TableFlags_.sortable |
            imgui.TableFlags_.resizable
        )
        if imgui.begin_table("##timings", 5, flags):
            imgui.table_setup_column("Extension")
            imgui.table_setup_column("Total", imgui.TableColumnFlags_.default_sort)
            imgui.table_setup_column("Update")
            imgui.table_setup_column("Event")
            imgui.table_setup_column("Render")
            imgui.table_headers_row()

            # Collect averages
            rows = []
            for name, samples in self._history.items():
                if not samples:
                    continue
                n = len(samples)
                avg_update = sum(s.update_ms for s in samples) / n
                avg_event = sum(s.event_ms for s in samples) / n
                avg_render = sum(s.render_ms for s in samples) / n
                avg_total = avg_update + avg_event + avg_render
                windows = {}
                for s in samples:
                    for wname, wms in s.windows.items():
                        if wname not in windows:
                            windows[wname] = []
                        windows[wname].append(wms)
                avg_windows = {
                    wname: sum(wms) / len(wms)
                    for wname, wms in windows.items()
                }
                rows.append((name, avg_total, avg_update, avg_event, avg_render, avg_windows))

            # Sort by total descending
            sort_specs = imgui.table_get_sort_specs()
            if sort_specs and sort_specs.specs_count > 0:
                col = sort_specs.get_specs(0).column_index
                ascending = sort_specs.get_specs(0).sort_direction == imgui.SortDirection.ascending
                rows.sort(key=lambda r: r[col + 1] if col > 0 else r[0], reverse=not ascending)
            else:
                rows.sort(key=lambda r: r[1], reverse=True)

            for name, total, update, event, render, avg_windows in rows:
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                is_open = imgui.tree_node(name)
                imgui.table_set_column_index(1)
                self._colored_ms(total)
                imgui.table_set_column_index(2)
                imgui.text(f"{update:.2f}")
                imgui.table_set_column_index(3)
                imgui.text(f"{event:.2f}")
                imgui.table_set_column_index(4)
                imgui.text(f"{render:.2f}")

                if is_open:
                    for wname, wms in sorted(avg_windows.items(), key=lambda x: -x[1]):
                        imgui.table_next_row()
                        imgui.table_set_column_index(0)
                        imgui.text(f"  {wname}")
                        imgui.table_set_column_index(1)
                        self._colored_ms(wms)
                    imgui.tree_pop()

            imgui.end_table()
        imgui.end()

    def _colored_ms(self, ms):
        if ms > 5.0:
            col = imgui.ImVec4(1.0, 0.3, 0.3, 1.0)
        elif ms > 2.0:
            col = imgui.ImVec4(1.0, 0.8, 0.2, 1.0)
        else:
            col = imgui.ImVec4(0.5, 0.5, 0.5, 1.0)
        imgui.text_colored(col, f"{ms:.2f}")
