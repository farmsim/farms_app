""" Widget interface

Widgets are pure UI components — they own no simulation state.
State is passed in via keyword arguments each frame; user actions
are reported via optional callbacks assigned on the instance.

"""

from typing import Callable, Optional
from imgui_bundle import imgui, em_to_vec2
from imgui_bundle.immapp import icons_fontawesome_6 as fa6


# ── Toolbar icons (FontAwesome 6) ──────────────────────────────────────────
class _Icons:
    STEP_BACK_MANY  = fa6.ICON_FA_BACKWARD_FAST + "##step_back_many"
    STEP_BACK       = fa6.ICON_FA_BACKWARD_STEP + "##step_back"
    PLAY            = fa6.ICON_FA_PLAY + "##play"
    PAUSE           = fa6.ICON_FA_PAUSE + "##pause"
    STOP            = fa6.ICON_FA_STOP + "##stop"
    STEP_FWD        = fa6.ICON_FA_FORWARD_STEP + "##step_fwd"
    STEP_FWD_MANY   = fa6.ICON_FA_FORWARD_FAST + "##step_fwd_many"
    RECORD          = fa6.ICON_FA_CIRCLE + "##record"


# ── Playback states ─────────────────────────────────────────────────────────
class PlaybackState:
    STOPPED   = "stopped"
    PLAYING   = "playing"
    PAUSED    = "paused"
    RECORDING = "recording"


class SimulationToolbar:
    """Transport controls toolbar for simulation panels.

    Pure UI — renders buttons and fires callbacks. Owns no simulation state.
    The owning extension is responsible for all state transitions.

    Usage
    -----
    In your extension's __init__::

        self.toolbar = SimulationToolbar()
        self.toolbar.on_play            = self._start_simulation
        self.toolbar.on_pause           = self._pause_simulation
        self.toolbar.on_stop            = self._stop_simulation
        self.toolbar.on_step_back       = lambda: self._step(-1)
        self.toolbar.on_step_back_many  = lambda: self._step(-10)
        self.toolbar.on_step_fwd        = lambda: self._step(1)
        self.toolbar.on_step_fwd_many   = lambda: self._step(10)
        self.toolbar.on_record          = self._toggle_record
        self.toolbar.on_speed_change    = self._set_speed

    In your extension's on_render::

        self.toolbar.render(
            playback_state=self.state,
            current_time=self.t,
            total_time=self.t_end,
            speed=self.speed,
        )

    Parameters
    ----------
    playback_state : str
        One of PlaybackState.STOPPED / PLAYING / PAUSED / RECORDING.
    current_time : float, optional
        Current simulation time in seconds.
    total_time : float, optional
        Total simulation duration in seconds. Pass None to hide timestamp.
    speed : float, optional
        Current playback speed multiplier (default 1.0).
    """

    # Available speed steps cycled by the speed button
    SPEED_STEPS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    def __init__(self):
        # ── Callbacks — assign to wire up behaviour ────────────────────────
        self.on_play:           Optional[Callable] = None
        self.on_pause:          Optional[Callable] = None
        self.on_stop:           Optional[Callable] = None
        self.on_step_back:      Optional[Callable] = None
        self.on_step_back_many: Optional[Callable] = None
        self.on_step_fwd:       Optional[Callable] = None
        self.on_step_fwd_many:  Optional[Callable] = None
        self.on_record:         Optional[Callable] = None
        self.on_speed_change:   Optional[Callable[[float], None]] = None
        self.on_scrub:          Optional[Callable[[float], None]] = None

        # Internal UI-only state
        self._tooltip_hovered: Optional[str] = None

    # ── Public render entry point ───────────────────────────────────────────
    def render(
        self,
        *,
        playback_state: str = PlaybackState.STOPPED,
        current_time:   Optional[float] = None,
        total_time:     Optional[float] = None,
        speed:          float = 1.0,
        recording:      bool = False,
        scrub_value:    float = 0.0,
        scrub_min:      float = 0.0,
        scrub_max:      float = 0.0,
    ) -> None:
        """Render the toolbar for one frame.

        Must be called inside an active imgui window (between begin/end).
        scrub_value/min/max control the timeline slider (e.g. -buffer_size..0).
        """
        is_playing   = playback_state == PlaybackState.PLAYING
        is_paused    = playback_state == PlaybackState.PAUSED
        is_recording = recording
        is_stopped   = playback_state == PlaybackState.STOPPED
        is_active    = is_playing

        style = imgui.get_style()
        btn_size = em_to_vec2(2, 1.6)

        # ── Navigation group: step back ────────────────────────────────────
        self._button(
            label=_Icons.STEP_BACK_MANY,
            size=btn_size,
            tooltip="Step back ×10",
            callback=self.on_step_back_many,
            disabled=is_stopped,
        )
        imgui.same_line(spacing=2)
        self._button(
            label=_Icons.STEP_BACK,
            size=btn_size,
            tooltip="Step back",
            callback=self.on_step_back,
            disabled=is_stopped,
        )

        # ── Separator ──────────────────────────────────────────────────────
        imgui.same_line(spacing=6)
        self._separator(height=btn_size.y)
        imgui.same_line(spacing=6)

        # ── Transport: play/pause toggle + stop ────────────────────────────
        if is_active:
            self._button(
                label=_Icons.PAUSE,
                size=btn_size,
                tooltip="Pause",
                callback=self.on_pause,
                active=True,
            )
        else:
            self._button(
                label=_Icons.PLAY,
                size=btn_size,
                tooltip="Play",
                callback=self.on_play,
            )
        imgui.same_line(spacing=2)
        self._button(
            label=_Icons.STOP,
            size=btn_size,
            tooltip="Stop",
            callback=self.on_stop,
            disabled=is_stopped,
        )

        # ── Separator ──────────────────────────────────────────────────────
        imgui.same_line(spacing=6)
        self._separator(height=btn_size.y)
        imgui.same_line(spacing=6)

        # ── Navigation group: step forward ─────────────────────────────────
        self._button(
            label=_Icons.STEP_FWD,
            size=btn_size,
            tooltip="Step forward",
            callback=self.on_step_fwd,
            disabled=is_stopped,
        )
        imgui.same_line(spacing=2)
        self._button(
            label=_Icons.STEP_FWD_MANY,
            size=btn_size,
            tooltip="Step forward ×10",
            callback=self.on_step_fwd_many,
            disabled=is_stopped,
        )

        # ── Separator ──────────────────────────────────────────────────────
        imgui.same_line(spacing=6)
        self._separator(height=btn_size.y)
        imgui.same_line(spacing=6)

        # ── Speed slider (steps in powers of 2: ..., ¼, ½, 1, 2, 4, ...) ─
        import math
        log_min, log_max = -3, 3  # 2^-3=⅛  to  2^3=8
        log_val = round(math.log2(max(0.125, speed)))
        log_val = max(log_min, min(log_max, log_val))
        label = f"1/{1 << -log_val}x" if log_val < 0 else f"{1 << log_val}x"
        imgui.set_next_item_width(em_to_vec2(7.0, 0).x)
        changed, new_log = imgui.slider_int(
            "##speed", log_val, log_min, log_max, label,
        )
        if changed and self.on_speed_change:
            self.on_speed_change(2.0 ** new_log)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Playback speed")

        # ── Separator ──────────────────────────────────────────────────────
        imgui.same_line(spacing=6)
        self._separator(height=btn_size.y)
        imgui.same_line(spacing=6)

        # ── Record button ──────────────────────────────────────────────────
        self._button(
            label=_Icons.RECORD,
            size=btn_size,
            tooltip="Record" if not is_recording else "Stop recording",
            callback=self.on_record,
            active=is_recording,
            danger=is_recording,
        )

        # ── Timeline scrub slider ────────────────────────────────────────
        if scrub_min < scrub_max:
            imgui.same_line(spacing=10)
            imgui.push_item_width(imgui.get_content_region_avail().x - 10)
            changed, new_val = imgui.slider_int(
                "##timeline", int(scrub_value),
                v_min=int(scrub_min), v_max=int(scrub_max),
                format="%d",
            )
            imgui.pop_item_width()
            if changed and self.on_scrub:
                self.on_scrub(new_val)

    # ── Private helpers ─────────────────────────────────────────────────────
    def _button(
        self,
        *,
        label:    str,
        size:     imgui.ImVec2,
        tooltip:  str,
        callback: Optional[Callable],
        active:   bool = False,
        danger:   bool = False,
        disabled: bool = False,
    ) -> None:
        """Render a single toolbar button with optional tinted active state."""
        style = imgui.get_style()

        if disabled:
            imgui.begin_disabled()

        if active:
            if danger:
                # Red tint for recording
                imgui.push_style_color(
                    imgui.Col_.button,
                    imgui.ImVec4(0.6, 0.1, 0.1, 0.4)
                )
                imgui.push_style_color(
                    imgui.Col_.button_hovered,
                    imgui.ImVec4(0.7, 0.15, 0.15, 0.5)
                )
            else:
                # Blue tint for play/pause active
                imgui.push_style_color(
                    imgui.Col_.button,
                    imgui.ImVec4(0.2, 0.4, 0.7, 0.4)
                )
                imgui.push_style_color(
                    imgui.Col_.button_hovered,
                    imgui.ImVec4(0.25, 0.45, 0.75, 0.5)
                )

        # Use label as imgui ID — strip display whitespace for uniqueness
        if imgui.button(label, size=size):
            if callback and not disabled:
                callback()

        if active:
            imgui.pop_style_color(2)

        if disabled:
            imgui.end_disabled()

        if imgui.is_item_hovered(imgui.HoveredFlags_.delay_short):
            imgui.set_tooltip(tooltip)

    def _separator(self, height: float) -> None:
        """Render a thin vertical separator aligned to button height."""
        cursor = imgui.get_cursor_screen_pos()
        draw_list = imgui.get_window_draw_list()
        col = imgui.get_color_u32(imgui.Col_.separator)
        padding = 4.0
        draw_list.add_line(
            imgui.ImVec2(cursor.x, cursor.y + padding),
            imgui.ImVec2(cursor.x, cursor.y + height - padding),
            col,
            thickness=1.0,
        )
        imgui.dummy(imgui.ImVec2(1, height))

    def _next_speed(self, current: float) -> float:
        """Cycle to the next speed step."""
        steps = self.SPEED_STEPS
        for i, s in enumerate(steps):
            if abs(s - current) < 1e-6:
                return steps[(i + 1) % len(steps)]
        return 1.0
