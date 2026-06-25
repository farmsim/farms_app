""" Describes colors palattes and helper functions to interact with colors """

import colorsys

from imgui_bundle import imgui


# Helper functions
def rgb_u32(r: float, g: float, b: float, a: float = 255.0):
    return imgui.color_convert_float4_to_u32((
        r / 255.0,
        g / 255.0,
        b / 255.0,
        a / 255.0,
    ))


# U32 -> float4
def u32_to_float4(col_u32):
    c = imgui.color_convert_u32_to_float4(col_u32)
    return (c.x, c.y, c.z, c.w)

# float4 tp U32
def float4_to_u32(col):
    return imgui.color_convert_float4_to_u32(col)


# Compute complementary color in HSV
# (fast, visually stable, theme-agnostic)
def complementary_u32(col_u32):
    r, g, b, a = u32_to_float4(col_u32)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    h_comp = (h + 0.5) % 1.0  # rotate hue by 180 degrees

    r2, g2, b2 = colorsys.hsv_to_rgb(h_comp, s, v)
    return float4_to_u32((r2, g2, b2, a))


# Light ↔ Dark switch by inverting value (brightness)
# keeps hue/saturation stable
def invert_theme_u32(col_u32):
    r, g, b, a = u32_to_float4(col_u32)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    v_inv = 1.0 - v * 0.85   # 85% preserves saturation better

    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v_inv)
    return float4_to_u32((r2, g2, b2, a))


_RAW_COLORS = {
    # Semantic neutrals
    "black":            (0, 0, 0),
    "white":            (255, 255, 255),
    "gray_light":       (220, 220, 220),
    "gray":             (140, 140, 140),
    "gray_dark":        (60, 60, 60),

    # Primaries
    "red":              (230, 90, 90),
    "green":            (120, 210, 140),
    "blue":             (130, 165, 255),

    # Highlights
    "yellow":           (255, 230, 120),
    "cyan":             (140, 240, 240),
    "magenta":          (240, 140, 240),

    # Pastels (publication style)
    "pastel_red":       (255, 185, 185),
    "pastel_blue":      (175, 205, 255),
    "pastel_green":     (185, 240, 200),
    "pastel_purple":    (215, 185, 255),
    "pastel_yellow":    (255, 245, 185),

    # Shadows / Highlights
    "shadow":           (0, 0, 0, 45),
    "shadow_soft":      (0, 0, 0, 30),
    "highlight":        (255, 255, 255, 50),
    "highlight_soft":   (255, 255, 255, 30),

    # Edges
    "edge":             (60, 60, 60),
    "edge_soft":        (120, 120, 120),
    "edge_strong":      (25, 25, 25),

    # Layer backgrounds
    "bg_light":         (245, 245, 250),
    "bg_neutral":       (230, 230, 240),
    "bg_accent":        (210, 220, 240),
}

# ------------------------------------------------------------
# Precompute all ImU32 colors once. Zero cost during rendering.
# ------------------------------------------------------------
COLORS = {name: rgb_u32(*rgba) for name, rgba in _RAW_COLORS.items()}

# freeze for safety
__all__ = ["COLORS"]
