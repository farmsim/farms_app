""" Visualize experimental data """
import time
import warnings

import cv2
import numpy as np
import OpenGL.GL as GL  # type: ignore
import pandas as pd
from farms_app.core.extension import CustomExtension, UIExtension
from farms_app.core.window import BaseWindow
from imgui_bundle import imgui, implot, implot3d
from imgui_bundle import portable_file_dialogs as pfd
from scipy.interpolate import PchipInterpolator
from scipy.signal import correlate


class VideoWindow(BaseWindow):
    """ Render Video  """

    def __init__(self, extension) -> None:
        name: str = "video"
        window_flags = (
            imgui.WindowFlags_.no_title_bar |
            imgui.WindowFlags_.no_resize |
            imgui.WindowFlags_.no_move |
            imgui.WindowFlags_.no_scrollbar |
            imgui.WindowFlags_.no_collapse |
            imgui.WindowFlags_.no_scroll_with_mouse |
            imgui.WindowFlags_.no_bring_to_front_on_focus |
            imgui.WindowFlags_.no_nav_focus
        )
        super().__init__(
            name=name,
            extension=extension,
            window_flags=window_flags,
            visible=True,
            dock_to_extension=False
        )

    def on_initialize(self):
        """ On initialize """

    def on_update(self):
        """ On update """
        pass

    def on_render(self):
        """ Render main extension dockspace """

        if self._extension.data:
            button_name = "Pause" if self._extension.play else "Play"
            if imgui.button(button_name):
                self._extension.play = not self._extension.play
            if self._extension.play:
                time.sleep(0.01)
            self._extension.seek_frame(self._extension.frame)
            if imgui.begin_child("Video", child_flags=imgui.ChildFlags_.borders | imgui.ChildFlags_.resize_x | imgui.ChildFlags_.resize_y):
                imgui.image(
                    int(self._extension.texture_id),
                    imgui.ImVec2((self._extension.frame_width, self._extension.frame_height)),
                    uv0=imgui.ImVec2((0,0)),
                    uv1=imgui.ImVec2((1,1)),
                    # border_color=imgui.ImVec4((1, 0, 0, 1))
                )
            imgui.end_child()


# ──────────────────────────────────────────────────────────────────────────────
# Pure-analysis helpers (no GUI dependencies)
# ──────────────────────────────────────────────────────────────────────────────

def _fill_missing(v: np.ndarray) -> np.ndarray:
    """Linear interpolation for interior NaNs, nearest-neighbour at edges."""
    v = v.copy().astype(float)
    if not np.any(np.isnan(v)):
        return v
    idx = np.arange(len(v))
    valid = ~np.isnan(v)
    if valid.sum() == 0:
        return v
    return np.interp(idx, idx[valid], v[valid])


def _pchip(values: np.ndarray, new_t: np.ndarray, n_orig: int) -> np.ndarray:
    """PCHIP upsample from [0..n_orig-1] onto new_t (0-based float indices)."""
    return PchipInterpolator(np.arange(n_orig, dtype=float), values, extrapolate=True)(new_t)


def _min_run_length(c: np.ndarray, min_len: int) -> np.ndarray:
    """Remove contact runs shorter than min_len frames."""
    c = np.array(c, dtype=bool).copy()
    d = np.diff(np.concatenate([[False], c, [False]]).astype(int))
    for on, off in zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1):
        if (off - on + 1) < min_len:
            c[on:off + 1] = False
    return c


def _build_passes(T_bc, n_frames: int, merge_gap: int, drop_leading: bool):
    """Return (passes [N,2], ss_evt [M]) using StartStop markers."""
    cols = {c.lower(): c for c in T_bc.columns}
    if 'startstop_x' not in cols:
        warnings.warn("StartStop_X missing – treating whole file as one pass.")
        return np.array([[0, n_frames - 1]]), np.array([], dtype=int)

    ss = ~T_bc[cols['startstop_x']].isna().values
    if 'startstop_y' in cols:
        ss = ss & ~T_bc[cols['startstop_y']].isna().values

    anim = np.interp(np.linspace(0, len(ss) - 1, n_frames),
                     np.linspace(0, len(ss) - 1, len(ss)),
                     ss.astype(float)) > 0.5
    idx = np.where(anim)[0]

    if len(idx) == 0:
        return np.array([[0, n_frames - 1]]), np.array([], dtype=int)

    if drop_leading and idx[0] <= 2:
        idx = idx[1:]
    if len(idx) == 0:
        return np.array([[0, n_frames - 1]]), np.array([], dtype=int)

    keep = np.concatenate([[True], np.diff(idx) > merge_gap])
    ss_evt = idx[keep]

    passes, p_start = [], 0
    for k in ss_evt:
        if int(k) >= p_start:
            passes.append([p_start, int(k)])
        p_start = int(k) + 1
    if p_start <= n_frames - 1:
        passes.append([p_start, n_frames - 1])
    return np.array(passes, dtype=int), ss_evt


def _cycle_lag_pct(a: np.ndarray, b: np.ndarray):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    xc = correlate(a, b, mode='full')
    lags = np.arange(-(n - 1), n)
    mask = np.abs(lags) <= n // 2
    xc, lags = xc[mask], lags[mask]
    norm = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if norm > 0:
        xc /= norm
    i = np.argmax(xc)
    return (lags[i] / (n - 1)) * 100, xc[i]


def _gait_template_text(g: str):
    tbl = {
        'pronk':            ('0',   '0',   '0'),
        'trot':             ('1/2', '1/2', '0'),
        'bound':            ('0',   '1/2', '1/2'),
        'halfbound':        ('0',   '1/3', '2/3'),
        'pace':             ('1/2', '0',   '1/2'),
        'canter':           ('2/3', '1/3', '0'),
        'rotarywalk':       ('3/4', '1/4', '1/2'),
        'rotarygallop':     ('3/4', '1/4', '1/2'),
        'transversewalk':   ('3/4', '1/2', '1/4'),
        'transversegallop': ('3/4', '1/2', '1/4'),
        'lateralseqwalk':   ('2/4', '1/4', '3/4'),
        'lateralseqamble':  ('2/4', '1/4', '3/4'),
        'diagonalseqwalk':  ('2/4', '3/4', '1/4'),
        'diagonalseqamble': ('2/4', '3/4', '1/4'),
    }
    return tbl.get(g.lower(), ('?', '?', '?'))


# Gait templates [hind L-R, homolateral, diagonal]
_TEMPLATE_NAMES = [
    'pronk', 'trot', 'bound', 'bound', 'pace',
    'halfbound', 'halfbound', 'canter', 'canter',
    'rotary', 'rotary', 'transverse', 'transverse',
    'lateral', 'diagonal',
]
_TEMPLATE_ARRAY = np.array([
    [0,    0,    0   ],
    [0.5,  0.5,  0   ],
    [0,    0.5,  0.5 ],
    [0,    2/3,  2/3 ],
    [0.5,  0,    0.5 ],
    [0,    1/3,  2/3 ],
    [0,    2/3,  1/3 ],
    [2/3,  1/3,  0   ],
    [1/3,  1/3,  2/3 ],
    [0.75, 0.25, 0.5 ],
    [0.25, 0.75, 0.5 ],
    [0.75, 0.5,  0.25],
    [0.25, 0.5,  0.75],
    [0.5,  0.25, 0.75],
    [0.5,  0.75, 0.25],
])


def _classify_gait(G: str) -> str:
    for token, label in [
        ('pronk', 'pronk'), ('trot', 'trot'), ('halfbound', 'halfbound'),
        ('bound', 'bound'), ('hop', 'hop'), ('canter', 'canter'),
        ('rotaryWalk', 'rotaryWalk'), ('rotaryGallop', 'rotaryGallop'),
        ('transverseWalk', 'transverseWalk'), ('transverseGallop', 'transverseGallop'),
        ('lateralSeqWalk', 'lateralSeqWalk'), ('lateralSeqAmble', 'lateralSeqAmble'),
        ('diagonalSeqWalk', 'diagonalSeqWalk'), ('diagonalSeqAmble', 'diagonalSeqAmble'),
        ('amble', 'amble'),
    ]:
        if token in G:
            return label
    return G


_PHASE_AXIS = np.linspace(0, 100, 101)

# Limb raster colours  (R, G, B, A) as 0-1 floats for implot
_RASTER_COLORS_IMPLOT = [
    (0.2, 0.6, 1.0, 1.0),   # LH – blue
    (0.2, 0.8, 0.2, 1.0),   # LF – green
    (1.0, 0.5, 0.1, 1.0),   # RF – orange
    (0.15, 0.15, 0.15, 1.0),# RH – near-black
]
_LIMB_ORDER  = [0, 2, 3, 1]   # display order: LH, LF, RF, RH
_LIMB_LABELS = ['LH', 'LF', 'RF', 'RH']

# Angle curve colours (L elbow, R elbow, L knee, R knee)
_ANGLE_COLORS = [
    (0.2,  0.45, 1.0,  1.0),
    (0.9,  0.2,  0.2,  1.0),
    (0.15, 0.7,  0.2,  1.0),
    (0.05, 0.05, 0.05, 1.0),
]


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis function  (called once on button press)
# ──────────────────────────────────────────────────────────────────────────────

def run_gait_analysis(
    data: dict,          # {"left": {col: array}, "right": {col: array}}
    bc_df,               # pandas DataFrame – bottom-cam CSV
    n_orig: int,         # number of original (raw) frames
    fps_raw: int  = 200,
    interp_f: int = 4,
    merge_gap: int = 3,
    drop_leading: bool = True,
    min_pass_sec: float = 0.25,
    min_contact_run: int = 3,
) -> dict:
    """
    Run the full LTKS gait pipeline on already-loaded data.

    Returns a `result` dict consumed by the viewer render methods:
      result['n_frames']        int
      result['contact_events']  list[np.ndarray(N,2)]  – 0-based frame indices
      result['cycle_start']     np.ndarray(nCycles,)
      result['cycle_end']       np.ndarray(nCycles,)
      result['gait_label']      list[str]
      result['phase']           np.ndarray(nCycles, 3)
      result['duty_factor']     np.ndarray(nCycles,)
      result['frame2cycle']     np.ndarray(n_frames,)  – 1-based cycle number (0 = none)
      result['Ang']             dict  gname -> {'mEL','sEL','mER','sER','mKL','sKL','mKR','sKR'}
      result['elbow_L/R']       np.ndarray(n_frames,)
      result['knee_L/R']        np.ndarray(n_frames,)
    """
    fps_up         = fps_raw * interp_f
    min_pass_frames = round(min_pass_sec * fps_up)

    joints = ['toe', 'ankle', 'knee', 'hip', 'iliac', 'shoulder', 'elbow', 'wrist', 'finger']

    # ── 1. Build coords dict with NaN fill + PCHIP upsample ────────────────
    coords = {'L': {}, 'R': {}}
    for j in joints:
        for side, key in [('L', 'left'), ('R', 'right')]:
            C = np.column_stack([
                _fill_missing(np.asarray(data[key].get(f'{j}_x', np.full(n_orig, np.nan)), float)),
                _fill_missing(np.asarray(data[key].get(f'{j}_y', np.full(n_orig, np.nan)), float)),
                _fill_missing(np.asarray(data[key].get(f'{j}_z', np.full(n_orig, np.nan)), float)),
            ])
            coords[side][j] = C

    # Mirror left-camera X & Z
    for j in joints:
        coords['L'][j][:, [0, 2]] *= -1

    # ── 2. X-lock + global Y-offset correction ─────────────────────────────
    L_iy = coords['L']['iliac'][:, 1]
    R_iy = coords['R']['iliac'][:, 1]
    global_Y_offset = np.nanmean(L_iy - R_iy)

    for f in range(n_orig):
        Lx = coords['L']['iliac'][f, 0]
        Rx = coords['R']['iliac'][f, 0]
        cx = (Lx + Rx) / 2
        for j in joints:
            coords['L'][j][f, 0] -= (Lx - cx)
            coords['R'][j][f, 0] -= (Rx - cx)
            coords['L'][j][f, 1] -= global_Y_offset / 2
            coords['R'][j][f, 1] += global_Y_offset / 2

    # ── 3. Upsample ────────────────────────────────────────────────────────
    new_frame_ct = (n_orig - 1) * interp_f + 1
    new_t = np.linspace(0, n_orig - 1, new_frame_ct)

    for j in joints:
        for side in ('L', 'R'):
            C = coords[side][j]
            coords[side][j] = np.column_stack([_pchip(C[:, d], new_t, n_orig) for d in range(3)])

    n_frames = new_frame_ct

    # ── 4. Angles ──────────────────────────────────────────────────────────
    def _ang(key, col):
        raw = np.asarray(data[key].get(col, np.full(n_orig, np.nan)), float)[:n_orig]
        return _pchip(_fill_missing(raw), new_t, n_orig)

    elbow_L = _ang('left',  'elbow_angle')
    elbow_R = _ang('right', 'elbow_angle')
    knee_L  = _ang('left',  'knee_angle')
    knee_R  = _ang('right', 'knee_angle')
    hip_L   = _ang('left',  'hip_angle')
    hip_R   = _ang('right', 'hip_angle')
    ankle_L = _ang('left',  'ankle_angle')
    ankle_R = _ang('right', 'ankle_angle')

    # ── 5. Iliac midpoint X velocity ───────────────────────────────────────
    midX = (coords['L']['iliac'][:, 0] + coords['R']['iliac'][:, 0]) / 2

    # ── 6. Pass windows ────────────────────────────────────────────────────
    passes, _ = _build_passes(bc_df, n_frames, merge_gap, drop_leading)
    lens = passes[:, 1] - passes[:, 0] + 1
    passes = passes[lens >= min_pass_frames]

    # Pass direction
    pass_dir = np.zeros(len(passes), dtype=int)
    for p in range(len(passes)):
        dx = np.diff(midX[passes[p, 0]: passes[p, 1] + 1])
        d = np.sign(np.nanmean(dx))
        pass_dir[p] = int(d) if d != 0 else 1

    # ── 7. Contact events per limb ─────────────────────────────────────────
    limbs = ['RearLeft', 'RearRight', 'FrontLeft', 'FrontRight']
    contact_events = []
    for limb in limbs:
        Xv = bc_df[f'{limb}_X'].values if f'{limb}_X' in bc_df.columns else np.full(len(bc_df), np.nan)
        Yv = bc_df[f'{limb}_Y'].values if f'{limb}_Y' in bc_df.columns else np.full(len(bc_df), np.nan)
        c_raw = (~np.isnan(Xv)) & (~np.isnan(Yv))
        c = np.interp(np.linspace(0, len(c_raw)-1, n_frames),
                      np.linspace(0, len(c_raw)-1, len(c_raw)),
                      c_raw.astype(float)) > 0.5
        c = _min_run_length(c, min_contact_run)
        d = np.diff(np.concatenate([[False], c, [False]]).astype(int))
        on  = np.where(d ==  1)[0]
        off = np.where(d == -1)[0] - 1
        contact_events.append(
            np.column_stack([on, off]) if len(on) else np.zeros((0, 2), dtype=int)
        )

    # ── 8. Cycle extraction ────────────────────────────────────────────────
    ref = contact_events[0]
    on_LH  = np.clip(ref[:, 0], 0, n_frames - 1) if len(ref) else np.array([], dtype=int)
    off_LH = np.clip(ref[:, 1], 0, n_frames - 1) if len(ref) else np.array([], dtype=int)

    c_start, c_end, stance_d, cycle_d, mid_ref, c_dir = [], [], [], [], [], []
    for pw in range(len(passes)):
        ps, pe = passes[pw]
        idx = np.where((on_LH >= ps) & (on_LH <= pe))[0]
        if len(idx) < 2:
            continue
        for ii in range(len(idx) - 1):
            s = on_LH[idx[ii]]
            e = on_LH[idx[ii + 1]] - 1
            if e > pe:
                continue
            offi   = min(off_LH[idx[ii]], e)
            stance = max(offi - s, 1)
            total  = max(e - s + 1, 1)
            c_start.append(s);  c_end.append(e)
            stance_d.append(stance); cycle_d.append(total)
            mid_ref.append(round(s + 0.5 * stance))
            c_dir.append(pass_dir[pw])

    cycle_start = np.array(c_start, dtype=int)
    cycle_end   = np.array(c_end,   dtype=int)
    stance_dur  = np.array(stance_d, dtype=float)
    cycle_dur   = np.array(cycle_d,  dtype=float)
    mid_ref     = np.array(mid_ref,  dtype=float)
    cycle_dir   = np.array(c_dir,    dtype=int)
    n_cycles    = len(cycle_start)
    duty_factor = np.where(cycle_dur > 0, stance_dur / cycle_dur, np.nan)

    # ── 9. Phase ───────────────────────────────────────────────────────────
    phase = np.full((n_cycles, 3), np.nan)
    for k in range(n_cycles):
        t0, t1 = cycle_start[k], cycle_end[k] + 1
        mids = [np.nan] * 4
        mids[0] = mid_ref[k]
        for L in range(1, 4):
            o = contact_events[L]
            if not len(o):
                continue
            mL   = o[:, 0] + 0.5 * (o[:, 1] - o[:, 0])
            cand = np.where((mL > t0) & (mL < t1))[0]
            if len(cand):
                mids[L] = mL[cand[0]]
        dur = cycle_dur[k]
        ph_LR = (mids[1] - mids[0]) / dur if not np.isnan(mids[1]) else np.nan
        ph_HL = (mids[2] - mids[0]) / dur if not np.isnan(mids[2]) else np.nan
        ph_DG = (mids[3] - mids[0]) / dur if not np.isnan(mids[3]) else np.nan
        if cycle_dir[k] < 0 and not np.isnan(ph_LR):
            ph_LR = -ph_LR
        phase[k, 0] = ph_LR % 1 if not np.isnan(ph_LR) else np.nan
        phase[k, 1] = ph_HL % 1 if not np.isnan(ph_HL) else np.nan
        phase[k, 2] = ph_DG % 1 if not np.isnan(ph_DG) else np.nan

    # ── 10. Gait classification ────────────────────────────────────────────
    gaits_all = []
    for k in range(n_cycles):
        D = np.abs(_TEMPLATE_ARRAY - phase[k])
        D[D > 0.5] = 1 - D[D > 0.5]
        scores = np.nansum(D ** 2, axis=1)
        gaits_all.append(_TEMPLATE_NAMES[int(np.argmin(scores))])

    # Duty-factor overrides
    _DF_OVERRIDES = {
        'bound':      ('hop',              'bound'),
        'halfbound':  ('hop',              'halfbound'),
        'rotary':     ('rotaryWalk',       'rotaryGallop'),
        'transverse': ('transverseWalk',   'transverseGallop'),
        'lateral':    ('lateralSeqWalk',   'lateralSeqAmble'),
        'diagonal':   ('diagonalSeqWalk',  'diagonalSeqAmble'),
    }
    for k in range(n_cycles):
        if gaits_all[k] in _DF_OVERRIDES and not np.isnan(duty_factor[k]):
            slow_name, fast_name = _DF_OVERRIDES[gaits_all[k]]
            gaits_all[k] = slow_name if duty_factor[k] >= 0.5 else fast_name

    gait_label = [_classify_gait(g) for g in gaits_all]

    # frame → cycle map (1-based, 0 = none)
    frame2cycle = np.zeros(n_frames, dtype=int)
    for k in range(n_cycles):
        frame2cycle[cycle_start[k]: cycle_end[k] + 1] = k + 1

    # ── 11. Phase-normalised angle analysis ────────────────────────────────
    def _resamp(v):
        return PchipInterpolator(np.linspace(0, 100, len(v)), v, extrapolate=True)(_PHASE_AXIS)

    Ang = {}
    gaits_present = list(dict.fromkeys(g for g in gait_label if g and g != 'nd'))
    for gname in gaits_present:
        idxC = [k for k, g in enumerate(gait_label) if g == gname]
        EL, ER, KL, KR = [], [], [], []
        for c in idxC:
            s, e = cycle_start[c], cycle_end[c]
            if e <= s:
                continue
            if cycle_dir[c] > 0:
                eL, eR = elbow_L[s:e+1], elbow_R[s:e+1]
                kL, kR = knee_L[s:e+1],  knee_R[s:e+1]
            else:
                eL, eR = elbow_R[s:e+1], elbow_L[s:e+1]
                kL, kR = knee_R[s:e+1],  knee_L[s:e+1]
            if len(eL) < 5:
                continue
            EL.append(_resamp(eL)); ER.append(_resamp(eR))
            KL.append(_resamp(kL)); KR.append(_resamp(kR))
        if not EL:
            continue
        EL = np.column_stack(EL); ER = np.column_stack(ER)
        KL = np.column_stack(KL); KR = np.column_stack(KR)
        nc = EL.shape[1]
        Ang[gname] = {
            'mEL': np.nanmean(EL, 1), 'sEL': 1.96 * np.nanstd(EL, 1) / np.sqrt(nc),
            'mER': np.nanmean(ER, 1), 'sER': 1.96 * np.nanstd(ER, 1) / np.sqrt(nc),
            'mKL': np.nanmean(KL, 1), 'sKL': 1.96 * np.nanstd(KL, 1) / np.sqrt(nc),
            'mKR': np.nanmean(KR, 1), 'sKR': 1.96 * np.nanstd(KR, 1) / np.sqrt(nc),
            'n':   nc,
        }

    # ── 12. Precompute 3D axis limits from processed coords ────────────────
    # Mirrors the MATLAB xlim_fixed / ylim_fixed / zlim_fixed block exactly.
    joints_list = list(coords['L'].keys())
    all_X = np.concatenate([coords[s][j][:, 0] for s in ('L', 'R') for j in joints_list])
    all_Y = np.concatenate([coords[s][j][:, 1] for s in ('L', 'R') for j in joints_list])
    all_Z = np.concatenate([coords[s][j][:, 2] for s in ('L', 'R') for j in joints_list])
    xlim = (float(np.nanmin(all_X)) - 1, float(np.nanmax(all_X)) + 1)
    ylim = (float(np.nanmin(all_Y)) - 1, float(np.nanmax(all_Y)) + 1)
    zlim = (float(np.nanmin(all_Z)) - 1, float(np.nanmax(all_Z)) + 1)

    return {
        'n_frames':       n_frames,
        'contact_events': contact_events,
        'cycle_start':    cycle_start,
        'cycle_end':      cycle_end,
        'gait_label':     gait_label,
        'phase':          phase,
        'duty_factor':    duty_factor,
        'frame2cycle':    frame2cycle,
        'Ang':            Ang,
        'elbow_L':        elbow_L,
        'elbow_R':        elbow_R,
        'knee_L':         knee_L,
        'knee_R':         knee_R,
        'hip_L':          hip_L,
        'hip_R':          hip_R,
        'ankle_L':        ankle_L,
        'ankle_R':        ankle_R,
        # Fully processed coords: mirrored, X-locked, Y-corrected, PCHIP-upsampled.
        # Shape: coords[side][joint] = np.ndarray(n_frames, 3)  with side in ('L','R')
        'coords':         coords,
        'xlim':           xlim,
        'ylim':           ylim,
        'zlim':           zlim,
    }


class KinematicsWindow(BaseWindow):
    """ Kinematics """

    # ── raster sliding-window width in frames ──
    _RASTER_WIN = 500

    # ── default trail length (ghost frames behind current frame) ──────────
    _TRAIL_LEN_DEFAULT = 50

    def __init__(self, extension) -> None:
        name: str = "kinematics"
        super().__init__(
            name=name,
            extension=extension,
            visible=True,
            dock_to_extension=False
        )
        self._analysis: dict | None = None      # populated after "Run Analysis"
        self._analysis_status: str  = ""        # status message shown in UI
        self._selected_gait_idx: int = 0        # combo selection for angle panel
        self._trail_len: int         = self._TRAIL_LEN_DEFAULT  # ghost frames

    def on_initialize(self):
        """ On initialize """
        pass

    def on_update(self):
        """ On update """
        pass

    def on_render(self):
        if not self._extension.data:
            imgui.text_disabled("No data loaded.")
            return

        if not self._analysis:
            if self._analysis_error:
                imgui.text_colored((1.0, 0.4, 0.4, 1.0), f"Analysis error: {self._analysis_error}")
            else:
                imgui.text_disabled("Running analysis…")
            return

        n_max = self._analysis['n_frames'] - 1

        # ── Transport bar (playback controls + scrub slider) ──────────────
        self._render_transport_bar(n_max)

        # Re-read frame — slider or buttons may have changed it this render
        frame = self._extension.frame

        if imgui.begin("Skeleton"):
            self._render_3d_skeleton(frame)
        imgui.end()
        if imgui.begin("Hind"):
            self._render_hind_trail(frame)
        imgui.end()
        if imgui.begin("Fore"):
            self._render_fore_trail(frame)
        imgui.end()
        if imgui.begin("Angles"):
            self._render_joint_angles(frame)
        imgui.end()
        if imgui.begin("Steps"):
            self._render_step_raster(frame)
        imgui.end()
        if imgui.begin("Phase"):
            self._render_phase_table(frame)
        imgui.end()
        if imgui.begin("Curves"):
            self._render_angle_curves()
        imgui.end()

    # ── sub-renderers ─────────────────────────────────────────────────────

    def _render_transport_bar(self, n_max: int):
        """
        Single-row transport bar:
          [⏮] [⏪] [▶/⏸] [⏩] [⏭]   |---|scrub-slider|---|   frame / total
        Playback auto-advance is handled here so it is always driven at render rate.
        """
        # ── Transport controls ─────────────────────────────────────────────
        frame = self._extension.frame

        # Use auto-sized buttons so they sit on one line
        if imgui.button("|<"):
            self._extension.frame = 0
            self._extension.play  = False
        imgui.same_line()

        # step back
        if imgui.button("<"):
            self._extension.frame = max(0, frame - 1)
            self._extension.play  = False
        imgui.same_line()

        # play-pause toggle
        play_label = "Pause" if self._extension.play else "Play"
        if imgui.button(play_label):
            self._extension.play = not self._extension.play
        imgui.same_line()

        # step forward
        if imgui.button(">"):
            self._extension.frame = min(n_max, frame + 1)
            self._extension.play  = False
        imgui.same_line()

        # jump to end
        if imgui.button(">|"):
            self._extension.frame = n_max
            self._extension.play  = False
        imgui.same_line()

        # Scrub slider — fills remaining width
        imgui.set_next_item_width(-100)   # leave room for "frame / total" text
        changed, new_frame = imgui.slider_int(
            "##scrub",
            self._extension.frame,
            0,
            n_max,
            format="",              # hide the default int label inside the slider
        )
        if changed:
            self._extension.frame = new_frame
            self._extension.play  = False   # scrubbing pauses playback
        imgui.same_line()
        imgui.text(f"{self._extension.frame} / {n_max}")

        # ── Auto-advance when playing ──────────────────────────────────────
        if self._extension.play:
            if self._extension.frame < n_max:
                self._extension.frame += 1
            else:
                self._extension.frame = 0   # loop

    def _run_analysis(self):
        self._analysis       = None
        self._analysis_error = ""
        try:
            data    = self._extension.data
            bc_df   = self._extension.bc_data
            n_left  = len(data['left'])
            n_right = len(data['right'])
            n_orig  = min(n_left, n_right)

            self._analysis = run_gait_analysis(
                data        = data,
                bc_df       = bc_df,
                n_orig      = n_orig,
                fps_raw     = getattr(self._extension, 'fps_raw',  200),
                interp_f    = getattr(self._extension, 'interp_f', 4),
            )
        except Exception as exc:
            self._analysis_error = str(exc)
            self._analysis       = None

    # ── 3D skeleton ───────────────────────────────────────────────────────

    # Segment chains, matching the MATLAB `connections` cell array order.
    # Each tuple is (proximal_joint, distal_joint).
    _HIND_CHAIN = ('iliac', 'hip', 'knee', 'ankle', 'toe')
    _FORE_CHAIN = ('shoulder', 'elbow', 'wrist', 'finger')

    # Camera-side → implot3d label prefix and line colour (RGBA 0-1).
    # L = left camera  (mirrored → blue),  R = right camera (red).
    # These match the MATLAB [0.2 0.4 1] / [1 0.3 0.3] colours.
    _SIDE_META = {
        'L': {'prefix': 'L',  'color': (0.2, 0.4, 1.0, 1.0)},
        'R': {'prefix': 'R',  'color': (1.0, 0.3, 0.3, 1.0)},
    }

    def _render_3d_skeleton(self, frame: int):
        """
        3D skeleton rendered from the fully-processed coords produced by
        run_gait_analysis (mirrored, X-locked, Y-corrected, PCHIP-upsampled).

        Before analysis has run the plot shows a disabled hint — raw extension
        data is NOT used here because left and right cameras are not yet aligned.

        Coordinate mapping follows the MATLAB script exactly:
          X = iliac midpoint progression axis
          Y = vertical (height)
          Z = lateral depth
        Axis limits are precomputed from the full processed trajectory range.
        """

        an     = self._analysis
        coords = an['coords']
        xlim   = an['xlim']
        ylim   = an['ylim']
        zlim   = an['zlim']

        # Gait label for title (1-based frame2cycle; 0 = between cycles)
        f2c   = an['frame2cycle']
        cyc   = int(f2c[frame]) if frame < len(f2c) else 0
        gait  = an['gait_label'][cyc - 1] if cyc > 0 else '—'
        title = f"Skeleton  |  Frame {frame}  |  Gait: {gait}"

        axes_flags = (
            implot.AxisFlags_.lock |
            implot.AxisFlags_.no_grid_lines |
            implot.AxisFlags_.no_tick_marks
        )

        if not implot3d.begin_plot("3D"):
            return

        # Axis setup — limits derived from actual processed data, not hardcoded.
        implot3d.setup_axes_limits(
            xlim[0], xlim[1],
            ylim[0], ylim[1],
            zlim[0], zlim[1],
        )
        implot3d.setup_box_scale(x=5.0, y=1.0, z=1.0)
        implot3d.setup_box_rotation(90.0, 0.0)   # top-down as in MATLAB view(ax,2)
        implot3d.setup_axis(implot3d.ImAxis3D_.x, label="X", flags=axes_flags)
        implot3d.setup_axis(implot3d.ImAxis3D_.y, label="Y", flags=axes_flags)
        implot3d.setup_axis(implot3d.ImAxis3D_.z, label="Z", flags=axes_flags)

        # Draw both camera sides using the processed, co-registered coords.
        for side, meta in self._SIDE_META.items():
            c     = coords[side]
            color = meta['color']
            pfx   = meta['prefix']

            # implot3d.push_style_color(implot3d.Col_.line, color)

            # Hind limb chain: iliac → hip → knee → ankle → toe
            implot3d.plot_line(
                f"{pfx}-hind",
                np.array([c[j][frame, 0] for j in self._HIND_CHAIN], dtype=np.float64),
                np.array([c[j][frame, 1] for j in self._HIND_CHAIN], dtype=np.float64),
                np.array([c[j][frame, 2] for j in self._HIND_CHAIN], dtype=np.float64),
            )

            # Fore limb chain: shoulder → elbow → wrist → finger
            implot3d.plot_line(
                f"{pfx}-fore",
                np.array([c[j][frame, 0] for j in self._FORE_CHAIN], dtype=np.float64),
                np.array([c[j][frame, 1] for j in self._FORE_CHAIN], dtype=np.float64),
                np.array([c[j][frame, 2] for j in self._FORE_CHAIN], dtype=np.float64),
            )

            # Spine connector: iliac → shoulder  (single segment)
            implot3d.plot_line(
                f"{pfx}-spine",
                np.array([c['iliac'][frame, 0], c['shoulder'][frame, 0]], dtype=np.float64),
                np.array([c['iliac'][frame, 1], c['shoulder'][frame, 1]], dtype=np.float64),
                np.array([c['iliac'][frame, 2], c['shoulder'][frame, 2]], dtype=np.float64),
            )

            # implot3d.pop_style_color()

        implot3d.end_plot()

    # ── trail helpers ─────────────────────────────────────────────────────────

    def _draw_trail_chain(
        self,
        plot_id: str,
        chain: tuple,
        frame: int,
        axes_flags: int,
    ) -> None:
        """
        Open a standalone implot3d plot and draw a fading ghost trail for `chain`
        across both camera sides.  Ghost alpha ramps from 0.05 (oldest) to 0.55
        (frame-1); current frame drawn at full opacity with a thicker line.
        Plot limits and initial rotation are shared with the main skeleton plot
        (Cond_.once so the user can rotate/zoom independently).
        """
        an     = self._analysis
        coords = an['coords']
        xlim, ylim, zlim = an['xlim'], an['ylim'], an['zlim']
        trail  = self._trail_len

        if not implot3d.begin_plot(plot_id):
            return

        implot3d.setup_axes_limits(
            xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1],
            cond=imgui.Cond_.once,
        )
        # implot3d.setup_box_initial_rotation(-85.0, 180.0)
        implot3d.setup_box_rotation(90.0, 0.0)   # top-down as in MATLAB view(ax,2)
        implot3d.setup_axis(implot3d.ImAxis3D_.x, label="X", flags=axes_flags)
        implot3d.setup_axis(implot3d.ImAxis3D_.y, label="Y", flags=axes_flags)
        implot3d.setup_axis(implot3d.ImAxis3D_.z, label="Z", flags=axes_flags)

        for side, meta in self._SIDE_META.items():
            c   = coords[side]
            r, g, b, _ = meta['color']
            pfx = meta['prefix']

            # Ghost frames — oldest to newest so newer ones paint on top
            for step in range(trail, 0, -1):
                gf    = max(0, frame - step)
                alpha = (0.05 + 0.50 * (1.0 - step / trail)) if trail > 0 else 0.0
                gs    = implot3d.Spec(line_color=(r, g, b, alpha), line_weight=1.0)
                implot3d.plot_line(
                    f"##{pfx}-g{step}",
                    np.array([c[j][gf, 0] for j in chain], dtype=np.float64),
                    np.array([c[j][gf, 1] for j in chain], dtype=np.float64),
                    np.array([c[j][gf, 2] for j in chain], dtype=np.float64),
                    spec=gs,
                )

            # Current frame
            cs = implot3d.Spec(line_color=(r, g, b, 1.0), line_weight=2.0)
            implot3d.plot_line(
                f"{pfx}",
                np.array([c[j][frame, 0] for j in chain], dtype=np.float64),
                np.array([c[j][frame, 1] for j in chain], dtype=np.float64),
                np.array([c[j][frame, 2] for j in chain], dtype=np.float64),
                spec=cs,
            )

        implot3d.end_plot()

    def _render_hind_trail(self, frame: int) -> None:
        """Standalone plot: hind limb trail (iliac → hip → knee → ankle → toe)."""
        axes_flags = implot.AxisFlags_.no_grid_lines | implot.AxisFlags_.no_tick_marks
        self._draw_trail_chain("Hind trail", self._HIND_CHAIN, frame, axes_flags)

    def _render_fore_trail(self, frame: int) -> None:
        """Standalone plot: fore limb trail (shoulder → elbow → wrist → finger)."""
        axes_flags = implot.AxisFlags_.no_grid_lines | implot.AxisFlags_.no_tick_marks
        self._draw_trail_chain("Fore trail", self._FORE_CHAIN, frame, axes_flags)

    # ── joint angle subplots ───────────────────────────────────────────────
    @staticmethod
    def _safe_slice(arr: np.ndarray, start: int, end: int) -> np.ndarray:
        """Return a contiguous 1-D float64 slice; empty array when length < 2."""
        s = np.ascontiguousarray(arr[start:end], dtype=np.float64).ravel()
        return s if s.size >= 2 else np.empty(0, dtype=np.float64)

    def _render_joint_angles(self, frame: int):
        """
        When analysis has run, use the PCHIP-upsampled angle arrays stored in
        self._analysis so they are time-synced with the upsampled frame index.
        Before analysis runs, fall back to raw data using the raw frame index
        (frame is still in the raw timeline at that point).
        """
        if not implot.begin_subplots("Joint angles", 3, 1, (-1, 300)):
            return

        win   = 100   # frames to show in the rolling window
        start = max(0, frame - win + 1)
        end   = frame + 1

        an    = self._analysis
        pairs = [
            ("left-hip",   an['hip_L'],   "right-hip",   an['hip_R']),
            ("left-knee",  an['knee_L'],  "right-knee",  an['knee_R']),
            ("left-ankle", an['ankle_L'], "right-ankle", an['ankle_R']),
        ]

        for left_label, left_arr, right_label, right_arr in pairs:
            if not implot.begin_plot(""):
                continue
            ls = self._safe_slice(left_arr,  start, end)
            rs = self._safe_slice(right_arr, start, end)
            if ls.size: implot.plot_line(left_label,  ls)
            if rs.size: implot.plot_line(right_label, rs)
            implot.end_plot()

        implot.end_subplots()

    # ── step raster ───────────────────────────────────────────────────────

    def _render_step_raster(self, frame: int):
        """
        Horizontal contact bars for the 4 limbs in a sliding ±250-frame window.
        Y axis rows (bottom→top): LH, LF, RF, RH.
        """
        an = self._analysis
        n  = an['n_frames']
        hw = self._RASTER_WIN // 2
        f_start = max(0, frame - hw)
        f_end   = min(n - 1, frame + (self._RASTER_WIN - hw - 1))

        if not implot.begin_plot("Step Raster", (-1, 120)):
            return

        implot.setup_axes("Frame", "", implot.AxisFlags_.none, implot.AxisFlags_.lock)
        implot.setup_axes_limits(f_start, f_end, 0.0, 4.0,
                                 imgui.Cond_.always)

        # Draw one thin horizontal bar per contact interval per limb
        for row_idx, limb_i in enumerate(_LIMB_ORDER):
            ev     = an['contact_events'][limb_i]
            color  = _RASTER_COLORS_IMPLOT[limb_i]
            y_bot  = float(row_idx) + 0.05
            y_top  = float(row_idx) + 0.85

            # implot.push_style_color(implot.Col_.fill, color)

            for ei in range(len(ev)):
                xs = int(ev[ei, 0])
                xe = int(ev[ei, 1])
                if xe < f_start:
                    continue
                if xs > f_end:
                    break
                xs_c = max(xs, f_start)
                xe_c = min(xe, f_end)
                if xe_c <= xs_c:
                    continue
                # Draw as a filled rectangle via two-point shaded region trick:
                # plot a vertical span using plot_shaded with xs..xe_c
                implot.plot_shaded(
                    _LIMB_LABELS[row_idx],
                    np.array([xs_c, xe_c], dtype=float),
                    np.array([y_bot, y_bot], dtype=float),
                    np.array([y_top, y_top], dtype=float),
                )

            # implot.pop_style_color()

        # Current-frame cursor
        # implot.push_style_color(implot.Col_.line, (1.0, 1.0, 0.2, 1.0))
        implot.plot_line("##cursor",
                         np.array([frame, frame], dtype=float),
                         np.array([0.0, 4.0], dtype=float))
        # implot.pop_style_color()

        # Custom Y tick labels
        # implot.set_next_axes_to_link(implot.ImAxis_.y1, 0)   # no-op guard
        # implot.setup_axis_ticks(
        #     implot.ImAxis_.y1,
        #     np.array([0.45, 1.45, 2.45, 3.45]),
        #     _LIMB_LABELS,
        #     keep_default=False,
        # )
        implot.end_plot()

    # ── per-cycle phase table ──────────────────────────────────────────────

    def _render_phase_table(self, frame: int):
        an = self._analysis
        f2c = an['frame2cycle']
        current_cyc = f2c[frame] if frame < len(f2c) else 0  # 1-based

        imgui.text("Phase values (relative to LH)")

        flags = (imgui.TableFlags_.borders_inner_v |
                 imgui.TableFlags_.borders_outer    |
                 imgui.TableFlags_.row_bg           |
                 imgui.TableFlags_.scroll_y         |
                 imgui.TableFlags_.sizing_fixed_fit)

        n_cyc = len(an['cycle_start'])
        row_height = imgui.get_text_line_height_with_spacing()

        if not imgui.begin_table(
            "##phase",
            6,
            flags,
            outer_size=imgui.ImVec2((0.0, row_height * min(n_cyc + 2, 8)))
        ):
            return

        for header in ("#", "Gait", "RH-LH", "LF-LH", "RF-LH", "Duty"):
            imgui.table_setup_column(header)
        imgui.table_headers_row()

        for k in range(n_cyc):
            imgui.table_next_row()

            # Highlight the cycle that contains the current frame
            if (k + 1) == current_cyc:
                imgui.table_set_bg_color(
                    imgui.TableBgTarget_.row_bg0,
                    imgui.color_convert_float4_to_u32((0.25, 0.55, 0.25, 0.45)),
                )

            imgui.table_set_column_index(0); imgui.text(str(k + 1))
            imgui.table_set_column_index(1); imgui.text(an['gait_label'][k])

            ph = an['phase'][k]
            df = an['duty_factor'][k]
            for col, val in enumerate([ph[0], ph[1], ph[2], df], start=2):
                imgui.table_set_column_index(col)
                imgui.text("—" if np.isnan(val) else f"{val:.3f}")

        imgui.end_table()

    # ── phase-normalised angle curves ─────────────────────────────────────

    def _render_angle_curves(self):
        an  = self._analysis
        Ang = an['Ang']
        if not Ang:
            imgui.text_disabled("No gait cycles detected for angle analysis.")
            return

        gnames = list(Ang.keys())

        # Gait selector combo
        imgui.text("Phase-normalised angles (mean ± 95 % CI)")
        changed, self._selected_gait_idx = imgui.combo(
            "##gait_select",
            self._selected_gait_idx,
            gnames,
        )
        self._selected_gait_idx = min(self._selected_gait_idx, len(gnames) - 1)
        gname = gnames[self._selected_gait_idx]
        g     = Ang[gname]
        LR, HL, DG = _gait_template_text(gname)
        n_c   = g['n']

        plot_title = f"{gname.upper()}  ({LR} {HL} {DG})  n={n_c} cycles"

        if not implot.begin_subplots("##angle_curves", 2, 1, (-1, 340)):
            return

        # ── Elbow panel ──
        if implot.begin_plot("Elbow"):
            implot.setup_axes("% cycle", "Angle (deg)")
            implot.setup_axes_limits(0, 100, 20, 160, imgui.Cond_.once)

            for mean_key, ci_key, color, label in [
                ('mEL', 'sEL', _ANGLE_COLORS[0], 'L Elbow'),
                ('mER', 'sER', _ANGLE_COLORS[1], 'R Elbow'),
            ]:
                m, s = g[mean_key], g[ci_key]
                # implot.push_style_color(
                #     implot.Col_.fill,
                #     (*color[:3], 0.25)
                # )
                implot.plot_shaded(f"##{label}_ci", _PHASE_AXIS,
                                    m - s, m + s)
                # implot.pop_style_color()
                # implot.push_style_color(implot.Col_.line, color)
                implot.plot_line(label, _PHASE_AXIS, m)
                # implot.pop_style_color()

            implot.end_plot()

        # ── Knee panel ──
        if implot.begin_plot("Knee"):
            implot.setup_axes("% cycle", "Angle (deg)")
            implot.setup_axes_limits(0, 100, 20, 160, imgui.Cond_.once)

            for mean_key, ci_key, color, label in [
                ('mKL', 'sKL', _ANGLE_COLORS[2], 'L Knee'),
                ('mKR', 'sKR', _ANGLE_COLORS[3], 'R Knee'),
            ]:
                m, s = g[mean_key], g[ci_key]
                # implot.push_style_color(implot.Col_.fill,
                #                          (*color[:3], 0.25))
                implot.plot_shaded(f"##{label}_ci", _PHASE_AXIS,
                                    m - s, m + s)
                # implot.pop_style_color()
                # implot.push_style_color(implot.Col_.line, color)
                implot.plot_line(label, _PHASE_AXIS, m)
                # implot.pop_style_color()

            implot.end_plot()

        implot.end_subplots()

    def _on_render(self):
        """ Render main extension dockspace """

        imgui.text("Hello!")

        if self._extension.data:

            _frame = self._extension.frame
            if implot3d.begin_plot("Mouse"):
                axes_flags = implot.AxisFlags_.lock | implot.AxisFlags_.no_grid_lines | implot.AxisFlags_.no_tick_marks
                implot3d.setup_box_scale(x=5.0, y=1.0, z=1.0)
                implot3d.setup_axes_limits(0, 800, -25.0, 160, 0, 100)
                implot3d.setup_box_initial_rotation(-85.0, 180.0)
                implot3d.setup_axis(implot3d.ImAxis3D_.x, label="x", flags=axes_flags)
                implot3d.setup_axis(implot3d.ImAxis3D_.y, label="y", flags=axes_flags)
                implot3d.setup_axis(implot3d.ImAxis3D_.z, label="z", flags=axes_flags)
                for side, data in self._extension.data.items():
                    # implot3d.set_next_marker_style(implot3d.Marker_.circle)
                    implot3d.plot_line(
                        f"{side}-hind",
                        np.array(
                            (
                                data["iliac_x"][_frame],
                                data["hip_x"][_frame],
                                data["knee_x"][_frame],
                                data["ankle_x"][_frame],
                                data["toe_x"][_frame],
                            )
                        ),
                        np.array(
                            (
                                data["iliac_y"][_frame],
                                data["hip_y"][_frame],
                                data["knee_y"][_frame],
                                data["ankle_y"][_frame],
                                data["toe_y"][_frame],
                            )
                        ),
                        np.array(
                            (
                                data["iliac_z"][_frame],
                                data["hip_z"][_frame],
                                data["knee_z"][_frame],
                                data["ankle_z"][_frame],
                                data["toe_z"][_frame],
                            )
                        ),
                    )
                    # implot3d.set_next_marker_style(implot3d.Marker_.circle)
                    implot3d.plot_line(
                        f"{side}-fore",
                        np.array(
                            (
                                data["shoulder_x"][_frame],
                                data["elbow_x"][_frame],
                                data["wrist_x"][_frame],
                                data["finger_x"][_frame],
                            )
                        ),
                        np.array(
                            (
                                data["shoulder_y"][_frame],
                                data["elbow_y"][_frame],
                                data["wrist_y"][_frame],
                                data["finger_y"][_frame],
                            )
                        ),
                        np.array(
                            (
                                data["shoulder_z"][_frame],
                                data["elbow_z"][_frame],
                                data["wrist_z"][_frame],
                                data["finger_z"][_frame],
                            )
                        ),
                    )
                implot3d.end_plot()

            if implot.begin_subplots("Joint angles", 3, 1, (-1, -1)):
                start_idx = max(0, _frame - 100 + 1)
                end_idx = _frame + 1
                if implot.begin_plot(""):
                    implot.plot_line("left-hip", np.array(self._extension.data["left"]["hip_angle"][start_idx:end_idx]))
                    implot.plot_line("right-hip", np.array(self._extension.data["right"]["hip_angle"][start_idx:end_idx]))
                    implot.end_plot()
                if implot.begin_plot(""):
                    implot.plot_line("left-knee", np.array(self._extension.data["left"]["knee_angle"][start_idx:end_idx]))
                    implot.plot_line("right-knee", np.array(self._extension.data["right"]["knee_angle"][start_idx:end_idx]))
                    implot.end_plot()
                if implot.begin_plot(""):
                    implot.plot_line("left-ankle", np.array(self._extension.data["left"]["ankle_angle"][start_idx:end_idx]))
                    implot.plot_line("right-ankle", np.array(self._extension.data["right"]["ankle_angle"][start_idx:end_idx]))
                    implot.end_plot()

                implot.end_subplots()

            if self._extension.frame < min(len(self._extension.data['left']), len(self._extension.data['right'])) - 1:
                if self._extension.play:
                    self._extension.frame += 1
            else:
                self._extension.frame = 0


class ExperimentalExtension(CustomExtension):
    """ Experimental """

    def __init__(self):

        name = "experimental"
        self.show_window = True
        self._io = imgui.get_io()
        self.data = {}
        self.frame = 0
        self.performance_warnings = []
        self.play = False

        self.current_frame = None
        self.texture_id = 0
        self.frame_width = 0
        self.frame_height = 0
        self.total_frames = 0
        self.current_frame_idx = 0
        self.fps = 30.0
        self.is_playing = False
        self.is_loaded = False
        self.last_frame_time = 0

        # File menu state — one async open_file dialog per channel
        self._dlg_L = None   # pfd.open_file for left CSV
        self._dlg_R = None   # pfd.open_file for right CSV
        self._dlg_BC = None   # pfd.open_file for bottom-cam CSV
        self._path_L: str    = ""     # last loaded path (display only)
        self._path_R: str    = ""
        self._path_BC: str    = ""
        self._load_error: str = ""     # non-empty on last load failure

        self.video_path = "/Users/tatarama/Downloads/LTKS-9152022-WK4-8-C3-cropped.mp4"
        self.cap = None
        super().__init__(name=name)
        self.register_window(VideoWindow(self))
        self.register_window(KinematicsWindow(self))

    def get_name(self) -> str:
        return "Experimental Data Viewer"

    def update_texture(self):
        """Update OpenGL texture with current frame"""
        if self.current_frame is None:
            return

        # Generate texture if not exists
        if self.texture_id == 0:
            self.texture_id = GL.glGenTextures(1)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                       self.frame_width, self.frame_height, 0,
                       GL.GL_RGB, GL.GL_UNSIGNED_BYTE, self.current_frame)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def load_video(self):
        """Load a video file"""
        try:
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"Error: Could not open video {self.video_path}")
                return False

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.current_frame_idx = 0
            self.is_loaded = True

            # Read first frame
            self.read_frame()

            print(f"Loaded video: {self.video_path}")
            print(f"Resolution: {self.frame_width}x{self.frame_height}")
            print(f"FPS: {self.fps}, Total frames: {self.total_frames}")

            return True

        except Exception as e:
            print(f"Error loading video: {e}")
            return False

    def read_frame(self):
        """Read current frame from video"""
        if not self.cap or not self.is_loaded:
            return False

        ret, frame = self.cap.read()
        if ret:
            # Convert BGR to RGB for OpenGL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = frame_rgb
            self.update_texture()
            return True
        return False

    def seek_frame(self, frame_idx):
        """Seek to specific frame"""
        if not self.is_loaded or not self.cap:
            return

        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.current_frame_idx = frame_idx
        self.read_frame()

    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        if self.texture_id != 0:
            GL.glDeleteTextures([self.texture_id])

    def on_initialize(self):
        """ On initialize """

    def get_dependencies(self):
        """ Get dependencies """

    def on_update(self):
        """ On update """

    def on_render(self):
        """ Render main extension dockspace """
        # Menu bar — always rendered first
        if not self.data:
            imgui.text_disabled("No data loaded — use File menu to open L / R / BC CSVs.")
            return

    def render_menu(self):
        """ Render menu """

        # ── Poll open-file dialogs ─────────────────────────────────────────
        for attr, side, key in [
            ("_dlg_L",  "left",  "L"),
            ("_dlg_R",  "right", "R"),
            ("_dlg_BC", None,    "BC"),
        ]:
            dlg = getattr(self, attr)
            if dlg is not None and dlg.ready():
                paths = dlg.result()          # list of selected paths
                setattr(self, attr, None)
                if paths:
                    self._load_csv(paths[0], side, key)

        if not imgui.begin_menu_bar():
            return

        if imgui.begin_menu("File"):
            # ── Open items ────────────────────────────────────────────────
            for label, attr, title, side, key in [
                ("Open Left CSV",       "_dlg_L",  "Open left kinematics CSV",  "left",  "L"),
                ("Open Right CSV",      "_dlg_R",  "Open right kinematics CSV", "right", "R"),
                ("Open Bottom-cam CSV", "_dlg_BC", "Open bottom-cam CSV",       None,    "BC"),
            ]:
                # Show a checkmark when that file is already loaded
                loaded = bool(getattr(self, f"_path_{key}"))
                display = f"[v] {label}" if loaded else f"    {label}"
                clicked = imgui.menu_item_simple(display)
                if clicked:
                    setattr(self, attr,
                            pfd.open_file(title, filters=["CSV files", "*.csv"]))

            imgui.separator()

            # ── Run Analysis ──────────────────────────────────────────────
            all_loaded = bool(self._path_L and self._path_R and self._path_BC
                              and self.data)
            if not all_loaded:
                # Grey out when files aren't ready
                imgui.begin_disabled()
            clicked = imgui.menu_item_simple("Run Analysis")

            if not all_loaded:
                imgui.end_disabled()

            if clicked and all_loaded:
                # self.windows[2].initialize()
                # self.windows[2]._run_analysis()
                pass

            imgui.end_menu()

        # Show load error in menu bar if present
        if self._load_error:
            imgui.same_line(spacing=20)
            imgui.text_colored((1.0, 0.5, 0.3, 1.0), f"Error: {self._load_error}")

        imgui.end_menu_bar()

    def _load_csv(self, path: str, side: str | None, key: str) -> None:
        """Load a single CSV file into the extension for the given channel."""

        self._load_error = ""
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            self._load_error = f"{key}: {exc}"
            return

        setattr(self, f"_path_{key}", path)

        if key in ("L", "R"):
            col_dict = {c: df[c].values for c in df.columns}
            if not self.data:
                self.data = {}
            self.data[side] = df
        else:  # BC
            self.bc_data = df

        # Reset playback and clear stale analysis whenever any file changes
        self.windows[2].hide = True
        self.frame = 0
        self.play  = False
        self._analysis        = None
        self._analysis_error  = ""
        self._selected_gait_idx = 0
