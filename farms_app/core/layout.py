""" Layout utilities.

RectCut implementation for computing layout geometry:

    r = viewport_rect()
    left = r.cut_left(200)   # 200px left panel; r is now the remainder
    bottom = r.cut_bottom(150)
    # r is now the center region
"""

from dataclasses import dataclass

from imgui_bundle import imgui


@dataclass
class Rect:
    """Axis-aligned rectangle used for layout calculations.

    Cut methods remove a strip from one edge and return it as a new Rect,
    shrinking self in-place. This lets you carve up a screen region without
    tracking offsets manually:

        r = viewport_rect()
        sidebar = r.cut_left(200)   # sidebar is 200px; r lost those pixels
        toolbar = r.cut_top(32)     # toolbar is 32px tall from what remains
        # r is now the center content area
    """
    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def width(self) -> float:
        return self.maxx - self.minx

    @property
    def height(self) -> float:
        return self.maxy - self.miny

    def cut_left(self, a: float) -> "Rect":
        """Cut a strip from the left edge. Returns the strip; self shrinks."""
        minx = self.minx
        self.minx = min(self.maxx, self.minx + a)
        return Rect(minx, self.miny, self.minx, self.maxy)

    def cut_right(self, a: float) -> "Rect":
        """Cut a strip from the right edge. Returns the strip; self shrinks."""
        maxx = self.maxx
        self.maxx = max(self.minx, self.maxx - a)
        return Rect(self.maxx, self.miny, maxx, self.maxy)

    def cut_top(self, a: float) -> "Rect":
        """Cut a strip from the top edge. Returns the strip; self shrinks."""
        miny = self.miny
        self.miny = min(self.maxy, self.miny + a)
        return Rect(self.minx, miny, self.maxx, self.miny)

    def cut_bottom(self, a: float) -> "Rect":
        """Cut a strip from the bottom edge. Returns the strip; self shrinks."""
        maxy = self.maxy
        self.maxy = max(self.miny, self.maxy - a)
        return Rect(self.minx, self.maxy, self.maxx, maxy)


def viewport_rect() -> Rect:
    """Return the current main viewport as a Rect."""
    vp = imgui.get_main_viewport()
    return Rect(vp.pos.x, vp.pos.y,
                vp.pos.x + vp.size.x, vp.pos.y + vp.size.y)
