# tinydraw.py
from __future__ import annotations
from typing import List, Tuple, Sequence
from PIL import Image, ImageDraw, ImageFilter
import numpy as np, math

Color = Tuple[int, int, int, int]  # RGBA 0..255

# ---------------------------
# Small utilities
# ---------------------------
def _rgba(c: Sequence[int]) -> Color:
    """Accept (r,g,b) or (r,g,b,a) -> RGBA"""
    if len(c) == 3:
        r, g, b = c
        return (int(r), int(g), int(b), 255)
    r, g, b, a = c
    return (int(r), int(g), int(b), int(a))

def _to_np(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.uint8)

def _from_np(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr, mode="RGBA")

def _soft_disk(diam: int, hardness: float) -> Image.Image:
    """Round brush mask: hardness∈[0,1], 1=hard edge."""
    d = max(1, int(diam))
    m = Image.new("L", (d, d), 0)
    ImageDraw.Draw(m).ellipse((0, 0, d-1, d-1), fill=255)
    if hardness < 1.0:
        blur = max(1, int((1.0 - hardness) * d * 0.5))
        m = m.filter(ImageFilter.GaussianBlur(blur))
    return m

def _stroke_mask(points: Sequence[Tuple[float, float]],
                 size: int,
                 hardness: float,
                 spacing: float = 0.15) -> Tuple[Image.Image, Tuple[int,int]]:
    """Stamp a soft brush along a polyline into a local mask; returns (mask, top-left offset)."""
    if not points:
        return Image.new("L", (1, 1), 0), (0, 0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ox = int(min(xs) - size)
    oy = int(min(ys) - size)
    w  = int(max(1, (max(xs) - min(xs)) + size * 2))
    h  = int(max(1, (max(ys) - min(ys)) + size * 2))
    mask = Image.new("L", (w, h), 0)
    brush = _soft_disk(size, hardness)
    bx, by = brush.size

    def stamp(x: float, y: float):
        mask.paste(brush, (int(x - ox - bx // 2), int(y - oy - by // 2)), brush)

    # sample along each segment
    for i in range(len(points) - 1):
        (x1, y1), (x2, y2) = points[i], points[i+1]
        dx, dy = x2 - x1, y2 - y1
        dist = max(1, int(math.hypot(dx, dy)))
        step = max(1, int(size * spacing))
        n = max(1, dist // step)
        for t in range(n + 1):
            u = t / n if n > 0 else 0.0
            stamp(x1 + dx * u, y1 + dy * u)

    if len(points) == 1:
        stamp(points[0][0], points[0][1])
    return mask, (ox, oy)

def _flood_fill(img: Image.Image, x: int, y: int, new_color: Color, tol: int = 12) -> Image.Image:
    """RGBA tolerance flood fill (4-neighborhood)."""
    arr = _to_np(img)
    H, W = arr.shape[:2]
    x = max(0, min(W - 1, int(x)))
    y = max(0, min(H - 1, int(y)))
    target = arr[y, x].astype(int)
    if np.all(np.abs(target - np.array(new_color)) <= 1):
        return img  # already that color

    seen = np.zeros((H, W), dtype=bool)
    q = [(x, y)]
    while q:
        cx, cy = q.pop()
        if seen[cy, cx]:
            continue
        seen[cy, cx] = True
        if np.all(np.abs(arr[cy, cx].astype(int) - target) <= tol):
            arr[cy, cx] = new_color
            if cx > 0:     q.append((cx - 1, cy))
            if cx < W - 1: q.append((cx + 1, cy))
            if cy > 0:     q.append((cx, cy - 1))
            if cy < H - 1: q.append((cx, cy + 1))
    return _from_np(arr)

# ---------------------------
# Public helpers (optional) the LLM can call to generate points
# ---------------------------
def line_points(x1: float, y1: float, x2: float, y2: float, step: int = 6) -> List[Tuple[float,float]]:
    """Sample a straight line as points (used with stroke)."""
    dist = max(1, int(math.hypot(x2 - x1, y2 - y1)))
    n = max(1, dist // max(1, step))
    return [(x1 + (x2 - x1) * t / n, y1 + (y2 - y1) * t / n) for t in range(n + 1)]

def circle_points(cx: float, cy: float, r: float, steps: int = 96) -> List[Tuple[float,float]]:
    """Sample a circle as a closed polyline (used with stroke)."""
    pts = []
    for i in range(steps + 1):
        a = 2 * math.pi * i / steps
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts

def bezier_points(p0, p1, p2, p3, steps: int = 64) -> List[Tuple[float,float]]:
    """Sample a cubic Bezier curve as points (used with stroke)."""
    pts = []
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        x = (u**3) * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + (t**3) * p3[0]
        y = (u**3) * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + (t**3) * p3[1]
        pts.append((x, y))
    return pts

# ---------------------------
# TinyDraw class
# ---------------------------
class TinyDraw:
    """
    Minimal creative drawing API:
      - set_brush(color, size, hardness)
      - stroke(points, erase=False)   # freehand soft brush
      - fill(x, y, tolerance=12)      # bucket fill
      - new_layer(), select_layer(), undo(), redo()
      - image() / save(path)

    Design: everything is drawn via strokes & fills (like MS Paint).
    """

    # Soft budgets (enforced to keep LLM outputs tidy)
    MAX_OPS_PER_IMAGE = 25
    MAX_POINTS_PER_STROKE = 200
    MAX_LAYERS = 4

    def __init__(self, w: int = 512, h: int = 512, bg: Color = (255, 255, 255, 255)):
        self.w, self.h = int(w), int(h)
        self.layers: List[Image.Image] = [Image.new("RGBA", (self.w, self.h), bg)]
        self._active = 0
        self._hist: List[List[Image.Image]] = []
        self._redo: List[List[Image.Image]] = []
        # brush state
        self.brush_color: Color = (0, 0, 0, 255)
        self.brush_size: int = 10
        self.brush_hardness: float = 0.9
        # accounting
        self._op_count = 0

    # -------- core state mgmt --------
    def _push(self) -> None:
        self._hist.append([ly.copy() for ly in self.layers])
        self._redo.clear()

    def undo(self) -> None:
        if self._hist:
            self._redo.append(self.layers)
            self.layers = self._hist.pop()

    def redo(self) -> None:
        if self._redo:
            self._hist.append(self.layers)
            self.layers = self._redo.pop()

    def _bump(self) -> None:
        self._op_count += 1
        # Soft guard; you can raise/disable if you prefer.
        if self._op_count > self.MAX_OPS_PER_IMAGE:
            # No hard exception to avoid crashing; you can change to raise.
            pass

    # -------- public API --------
    def set_brush(self,
                  color: Sequence[int] = (0, 0, 0, 255),
                  size: int = 10,
                  hardness: float = 0.9) -> None:
        """Set current brush color/size/hardness."""
        self.brush_color = _rgba(color)
        self.brush_size = max(1, int(size))
        self.brush_hardness = float(max(0.0, min(1.0, hardness)))
        self._bump()

    def new_layer(self) -> None:
        """Add a transparent layer and select it."""
        if len(self.layers) >= self.MAX_LAYERS:
            return
        self._push(); self._bump()
        self.layers.append(Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0)))
        self._active = len(self.layers) - 1

    def select_layer(self, idx: int) -> None:
        """Select existing layer index."""
        if 0 <= idx < len(self.layers):
            self._active = idx
            self._bump()

    def stroke(self, points: Sequence[Tuple[float, float]], erase: bool = False) -> None:
        """Draw a freehand polyline with the current soft brush."""
        if not points:
            return
        if len(points) > self.MAX_POINTS_PER_STROKE:
            points = points[:self.MAX_POINTS_PER_STROKE]  # trim
        mask, (ox, oy) = _stroke_mask(points, self.brush_size, self.brush_hardness)
        self._push(); self._bump()
        layer = self.layers[self._active]
        # prepare stamp (ink or erase)
        stamp = Image.new("RGBA", mask.size, (0, 0, 0, 0) if erase else self.brush_color)
        stamp.putalpha(mask)
        tmp = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        tmp.paste(stamp, (ox, oy), stamp)
        self.layers[self._active] = Image.alpha_composite(layer, tmp)

    def fill(self, x: int, y: int, tolerance: int = 12) -> None:
        """Bucket fill with current brush color and tolerance."""
        self._push(); self._bump()
        self.layers[self._active] = _flood_fill(self.layers[self._active], x, y, self.brush_color, tol=int(tolerance))

    # -------- outputs --------
    def image(self) -> Image.Image:
        """Flatten layers and return a PIL Image (RGBA)."""
        out = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        for ly in self.layers:
            out = Image.alpha_composite(out, ly)
        return out
