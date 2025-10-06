# bicycle.py
# Draws a stylized bicycle using only Python's standard turtle module.

from PIL import Image
import turtle as T
import math

# ---------- basic helpers ----------
def jump(x, y):
    """Move without drawing."""
    T.penup()
    T.goto(x, y)
    T.pendown()

def line_to(x, y, width=3):
    T.width(width)
    T.goto(x, y)

def draw_circle(center, r, color="black", fill=None, width=3, steps=120):
    cx, cy = center
    T.width(width)
    T.color(color, fill or "")
    jump(cx, cy - r)
    if fill:
        T.begin_fill()
    T.circle(r, steps=steps)
    if fill:
        T.end_fill()

def draw_ring(center, r_outer, r_inner, outline="black", fill="black", width=2):
    # outer
    draw_circle(center, r_outer, color=outline, fill=fill, width=width)
    # inner "hole"
    T.color("white", "white")
    draw_circle(center, r_inner, color="white", fill="white", width=1)

def draw_spokes(center, r, count=14, color="#404040", width=2):
    cx, cy = center
    T.color(color)
    T.width(width)
    for i in range(count):
        ang = 2 * math.pi * i / count
        x = cx + r * math.cos(ang)
        y = cy + r * math.sin(ang)
        jump(cx, cy)
        line_to(x, y, width)

def draw_arc(center, r, start_deg, extent_deg, width=6, color="#202020"):
    cx, cy = center
    T.width(width)
    T.color(color)
    # Turtle draws arcs relative to current heading from the arc start point.
    # Convert to starting point and heading.
    start_rad = math.radians(start_deg)
    sx = cx + r * math.cos(start_rad)
    sy = cy + r * math.sin(start_rad)
    jump(sx, sy)
    T.setheading(start_deg + 90)  # heading tangent to the circle
    T.circle(r, extent_deg)

# ---------- bicycle parameters ----------
SCALE = 1.0   # tweak to make the whole bike bigger/smaller
# Wheel geometry
R_WHEEL = 90 * SCALE
# Wheel centers (roughly horizontal)
REAR = (-170 * SCALE, -80 * SCALE)
FRONT = ( 170 * SCALE, -80 * SCALE)

# Frame key points (rough diamond frame)
BB      = (-70 * SCALE, -60 * SCALE)   # bottom bracket (crank center)
REAR_AX = REAR                          # rear axle
FRONT_AX= FRONT                         # front axle
SEAT_T  = (-30 * SCALE,  60 * SCALE)    # seat tube – top (seat cluster)
HEAD_T  = ( 80 * SCALE,  95 * SCALE)    # head tube – top
HEAD_B  = ( 60 * SCALE,  50 * SCALE)    # head tube – bottom
SEAT    = (-25 * SCALE, 100 * SCALE)    # saddle position
STEM    = ( 85 * SCALE, 105 * SCALE)    # stem/handlebar center

# Colors
COLOR_FRAME   = "#1f77b4"
COLOR_WHEEL   = "#111111"
COLOR_TIRE    = "#111111"
COLOR_RIM     = "#888888"
COLOR_SPOKE   = "#606060"
COLOR_CHAIN   = "#444444"
COLOR_SEAT    = "#222222"
COLOR_HANDLE  = "#222222"
COLOR_PEDAL   = "#333333"
BG            = "#f7f7fa"

# ---------- draw parts ----------
def draw_wheel(center):
    # Tire
    draw_ring(center, R_WHEEL, R_WHEEL - 14 * SCALE, outline=COLOR_TIRE, fill="#202020", width=3)
    # Rim
    draw_ring(center, R_WHEEL - 18 * SCALE, R_WHEEL - 24 * SCALE, outline=COLOR_RIM, fill="#cccccc", width=2)
    # Hub
    draw_ring(center, 8 * SCALE, 3 * SCALE, outline=COLOR_RIM, fill="#dddddd", width=1)
    # Spokes
    draw_spokes(center, R_WHEEL - 21 * SCALE, count=20, color=COLOR_SPOKE, width=int(1.5 * SCALE) or 1)

def draw_tube(a, b, width=10, color=COLOR_FRAME):
    T.color(color)
    T.width(width)
    jump(*a)
    line_to(*b, width)

def draw_frame():
    # Main triangle: rear axle -> BB -> head tube bottom -> back to seat cluster -> rear axle
    draw_tube(REAR_AX, BB)
    draw_tube(BB, HEAD_B)
    draw_tube(HEAD_B, SEAT_T)
    draw_tube(SEAT_T, REAR_AX)

    # Top tube: seat cluster -> head tube top
    draw_tube(SEAT_T, HEAD_T)

    # Seat tube: seat cluster -> bottom bracket
    draw_tube(SEAT_T, BB)

    # Down tube: head bottom -> bottom bracket
    draw_tube(HEAD_B, BB)

    # Head tube: top -> bottom
    draw_tube(HEAD_T, HEAD_B, width=12)

def draw_fork():
    # Fork blades: head bottom to front axle, as two slightly separated lines
    offset = 6 * SCALE
    # Compute slight offsets perpendicular to the blade direction
    angle = math.atan2(FRONT_AX[1] - HEAD_B[1], FRONT_AX[0] - HEAD_B[0])
    nx, ny = -math.sin(angle), math.cos(angle)
    hb1 = (HEAD_B[0] + nx * offset, HEAD_B[1] + ny * offset)
    hb2 = (HEAD_B[0] - nx * offset, HEAD_B[1] - ny * offset)
    fa1 = (FRONT_AX[0] + nx * offset, FRONT_AX[1] + ny * offset)
    fa2 = (FRONT_AX[0] - nx * offset, FRONT_AX[1] - ny * offset)
    draw_tube(hb1, fa1, width=8)
    draw_tube(hb2, fa2, width=8)

def draw_handlebar():
    T.color(COLOR_HANDLE)
    T.width(8)
    # Stem
    draw_tube(HEAD_T, STEM, width=8, color=COLOR_HANDLE)
    # Bar (slight curve via two straight segments)
    bar_span = 80 * SCALE
    left = (STEM[0] - bar_span, STEM[1] + 0 * SCALE)
    right = (STEM[0] + bar_span, STEM[1] + 0 * SCALE)
    draw_tube(left, right, width=6, color=COLOR_HANDLE)
    # Grips
    T.width(12)
    draw_arc(left, 12 * SCALE, 180, 180, width=10, color=COLOR_HANDLE)
    draw_arc(right, 12 * SCALE, 0, 180, width=10, color=COLOR_HANDLE)

def draw_seat():
    T.color(COLOR_SEAT)
    # Seat post
    draw_tube(SEAT_T, (SEAT[0], SEAT_T[1] + 5 * SCALE), width=6, color=COLOR_SEAT)
    # Saddle (simple capsule)
    w, h = 55 * SCALE, 10 * SCALE
    cx, cy = SEAT
    jump(cx - w/2, cy)
    T.setheading(0)
    T.width(10)
    T.color(COLOR_SEAT, COLOR_SEAT)
    T.begin_fill()
    for _ in range(2):
        T.forward(w)
        T.circle(h, 180)
    T.end_fill()

def draw_drivetrain():
    # Chainring & crank
    ring_r = 26 * SCALE
    draw_ring(BB, ring_r, ring_r - 6 * SCALE, outline=COLOR_CHAIN, fill="#b0b0b0", width=2)

    # Crank arms (two opposite)
    for ang in (30, 210):
        rad = math.radians(ang)
        end = (BB[0] + 45 * SCALE * math.cos(rad), BB[1] + 45 * SCALE * math.sin(rad))
        draw_tube(BB, end, width=6, color=COLOR_PEDAL)
        # Pedals
        ped = 16 * SCALE
        nx, ny = -math.sin(rad), math.cos(rad)
        p1 = (end[0] - nx * ped, end[1] - ny * ped)
        p2 = (end[0] + nx * ped, end[1] + ny * ped)
        draw_tube(p1, p2, width=8, color=COLOR_PEDAL)

    # Rear sprocket + chain line
    sprocket_r = 10 * SCALE
    draw_ring(REAR_AX, sprocket_r, sprocket_r - 4 * SCALE, outline=COLOR_CHAIN, fill="#d0d0d0", width=2)

    # Chain (two straight runs with rounded ends)
    T.color(COLOR_CHAIN)
    T.width(5)
    # Upper run
    jump(BB[0] + ring_r, BB[1])
    line_to(REAR_AX[0] + sprocket_r, REAR_AX[1])
    # Lower run
    jump(BB[0] + ring_r - 4 * SCALE, BB[1] - 8 * SCALE)
    line_to(REAR_AX[0] + sprocket_r - 4 * SCALE, REAR_AX[1] - 8 * SCALE)

def draw_brakes_and_details():
    # Seat stays and chain stays to the rear axle (thin)
    draw_tube(SEAT_T, REAR_AX, width=6, color=COLOR_FRAME)  # already drawn but this thickens nicely
    # Simple front brake arc on rim
    draw_arc(FRONT_AX, R_WHEEL - 22 * SCALE, 120, 30, width=4, color="#303030")
    # Simple rear brake arc
    draw_arc(REAR_AX, R_WHEEL - 22 * SCALE, 60, 30, width=4, color="#303030")

def draw_ground():
    # A subtle ground line under the wheels
    T.color("#bbbbbb")
    T.width(2)
    y = REAR[1] - (R_WHEEL + 18 * SCALE)
    jump(-400 * SCALE, y)
    line_to(400 * SCALE, y, 2)

# ---------- main ----------
def main():
    T.title("Turtle Bicycle")
    T.setup(width=900, height=600)
    T.bgcolor(BG)
    T.hideturtle()
    T.tracer(False)  # draw fast, then reveal

    # Wheels first
    draw_wheel(REAR)
    draw_wheel(FRONT)

    # Frame & components
    draw_frame()
    draw_fork()
    draw_handlebar()
    draw_seat()
    draw_drivetrain()
    draw_brakes_and_details()
    draw_ground()

    T.tracer(True)

    canvas = T.getcanvas()
    canvas.postscript(file="gpt.eps")
    img = Image.open("gpt.eps")
    img.save("gpt.png")

    T.done()

if __name__ == "__main__":
    main()
