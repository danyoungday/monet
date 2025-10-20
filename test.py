from turtle import Turtle
from PIL import Image
import math
import random

def draw_image(t: Turtle) -> None:
    """
    Draw a stylized, colorful cat using the turtle library.
    The drawing includes a decorative background, a patchwork body,
    expressive eyes, a heart-shaped nose, whiskers, a striped tail,
    and a little collar with a bell.
    """
    # Helper functions
    def move(x, y):
        t.penup()
        t.goto(x, y)
        t.pendown()

    def circle_center(x, y, r, fillcolor=None, pencolor=None, pensz=1):
        """
        Draw a circle with center at (x, y) and radius r.
        """
        if pencolor:
            t.pencolor(pencolor)
        t.pensize(pensz)
        if fillcolor:
            t.fillcolor(fillcolor)
            t.begin_fill()
        move(x, y - r)
        t.setheading(0)
        t.circle(r)
        if fillcolor:
            t.end_fill()

    def semi_circle_center(x, y, r, extent=180, fillcolor=None, pencolor=None, pensz=1, orientation=1):
        """
        Draw a semicircle (or arc) centered at (x, y) with radius r.
        orientation=1 draws the arc from left to right (top half); -1 flips it.
        extent in degrees.
        """
        if pencolor:
            t.pencolor(pencolor)
        t.pensize(pensz)
        if fillcolor:
            t.fillcolor(fillcolor)
            t.begin_fill()
        # place turtle at leftmost point if drawing top arc; adjust by orientation
        angle = 90 * orientation
        move(x + r * math.cos(math.radians(angle)), y + r * math.sin(math.radians(angle)))
        t.setheading(180 * (1 - orientation) / 2)  # roughly point tangentially
        t.circle(r * orientation, extent)
        if fillcolor:
            t.end_fill()

    def polygon(points, fillcolor=None, pencolor=None, pensz=1):
        if pencolor:
            t.pencolor(pencolor)
        t.pensize(pensz)
        if fillcolor:
            t.fillcolor(fillcolor)
            t.begin_fill()
        move(points[0][0], points[0][1])
        for (px, py) in points[1:]:
            t.goto(px, py)
        t.goto(points[0][0], points[0][1])
        if fillcolor:
            t.end_fill()

    def draw_background():
        # soft circle background behind the cat
        circle_center(0, 20, 360, fillcolor="#EAF6FF", pencolor="#EAF6FF")
        # scatter some stylized stars
        random.seed(42)
        star_colors = ["#FFF9C4", "#FFE0B2", "#FCE4EC", "#E1F5FE"]
        for i in range(20):
            sx = random.randint(-380, 380)
            sy = random.randint(-220, 320)
            col = random.choice(star_colors)
            t.pencolor(col)
            t.fillcolor(col)
            move(sx, sy)
            t.begin_fill()
            for _ in range(4):
                t.forward(6)
                t.right(90)
            t.end_fill()

    def draw_patch(x, y, w, h, angle=0, fill="#FFD59E"):
        # draw an oriented rectangle patch as polygon for patchwork
        hw = w / 2
        hh = h / 2
        # corners before rotation
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        pts = []
        for cx, cy in corners:
            rx = cx * math.cos(math.radians(angle)) - cy * math.sin(math.radians(angle)) + x
            ry = cx * math.sin(math.radians(angle)) + cy * math.cos(math.radians(angle)) + y
            pts.append((rx, ry))
        polygon(pts, fillcolor=fill, pencolor="#8C5A30", pensz=2)

    def draw_body():
        # Main rounded body (big circle)
        circle_center(0, -40, 140, fillcolor="#F6D6A8", pencolor="#8C5A30", pensz=3)

        # Patchwork: several patches with different patterns
        patches = [
            (-40, -20, 80, 60, -20, "#FFB6B9"),
            (50, -10, 90, 70, 15, "#C1E1C1"),
            (-10, -80, 120, 50, 10, "#FFD59E"),
            (30, -70, 60, 50, -25, "#E1F5FE"),
        ]
        for x, y, w, h, ang, col in patches:
            draw_patch(x, y, w, h, ang, col)
            # add polka dots inside patch
            for dx in range(3):
                for dy in range(2):
                    px = x + (dx - 1) * (w / 6) + random.uniform(-6, 6)
                    py = y + (dy - 0.5) * (h / 4) + random.uniform(-6, 6)
                    circle_center(px, py, 6, fillcolor="#FFFFFF", pencolor="#FFFFFF")

        # stripes on body
        t.pencolor("#D2945B")
        t.pensize(4)
        for i in range(5):
            move(-100 + i * 40, -20 + 12 * math.sin(i))
            t.setheading(320 + i * 5)
            t.circle(60, 40)

    def draw_head():
        # Head circle
        circle_center(0, 80, 80, fillcolor="#F7E0C6", pencolor="#8C5A30", pensz=3)

        # Ears - outer
        left_ear = [(-60, 150), (-20, 190), (-10, 150)]
        right_ear = [(60, 150), (20, 190), (10, 150)]
        polygon(left_ear, fillcolor="#F6D6A8", pencolor="#8C5A30", pensz=3)
        polygon(right_ear, fillcolor="#F6D6A8", pencolor="#8C5A30", pensz=3)
        # ear inner
        polygon([(-46, 152), (-22, 178), (-16, 153)], fillcolor="#FFD2E0", pencolor="#C77B8D", pensz=2)
        polygon([(46, 152), (22, 178), (16, 153)], fillcolor="#FFD2E0", pencolor="#C77B8D", pensz=2)

        # forehead stripe
        t.pencolor("#D2945B")
        t.pensize(3)
        move(-20, 125)
        t.setheading(60)
        t.circle(30, 120)

    def draw_eye(cx, cy, size=22, iris="#5DA1FF"):
        # white
        circle_center(cx, cy, size, fillcolor="#FFFFFF", pencolor="#6B6B6B", pensz=2)
        # iris
        circle_center(cx, cy + 3, size * 0.55, fillcolor=iris, pencolor=iris)
        # pupil
        circle_center(cx, cy + 5, size * 0.2, fillcolor="#000000", pencolor="#000000")
        # sparkle
        circle_center(cx + 6, cy + 11, size * 0.12, fillcolor="#FFFFFF", pencolor="#FFFFFF")

    def draw_eyes_and_nose():
        # Eyes
        draw_eye(-30, 95, size=26, iris="#6ABED6")
        draw_eye(30, 95, size=26, iris="#6ABED6")

        # Nose: stylized heart-like (two small semicircles + triangle)
        # left bump
        circle_center(-8, 69, 6, fillcolor="#FF8FA3", pencolor="#C85B6E", pensz=2)
        circle_center(8, 69, 6, fillcolor="#FF8FA3", pencolor="#C85B6E", pensz=2)
        # lower triangle
        polygon([( -8, 64 ), (8, 64), (0, 54)], fillcolor="#FF8FA3", pencolor="#C85B6E", pensz=2)

        # little mouth
        t.pencolor("#C85B6E")
        t.pensize(2)
        move(0, 54)
        t.setheading(-60)
        t.circle(12, 120)

    def draw_whiskers():
        t.pencolor("#6B6B6B")
        t.pensize(2)
        wlen = 80
        # three whiskers each side, slight curves using circle arcs
        for i, offset in enumerate([0, -8, 8]):
            # right side
            move(18, 66 + offset)
            t.setheading(350 - i * 6)
            t.circle(-60 - i * 6, 45)
            # left side
            move(-18, 66 + offset)
            t.setheading(190 + i * 6)
            t.circle(60 + i * 6, 45)

    def draw_paws():
        # front paws (two small overlapping circles at bottom of body)
        circle_center(-45, -160, 28, fillcolor="#F6D6A8", pencolor="#8C5A30", pensz=3)
        circle_center(45, -160, 28, fillcolor="#F6D6A8", pencolor="#8C5A30", pensz=3)
        # toe lines
        t.pencolor("#8C5A30")
        t.pensize(2)
        for px in (-45, 45):
            for i in range(3):
                move(px - 10 + i * 10, -160 + 8)
                t.setheading(80)
                t.forward(8)

    def draw_tail():
        # wavy tail made from sequential arcs with alternating colors (striped tail)
        base_x, base_y = 120, -10
        t.pensize(8)
        tail_colors = ["#F6D6A8", "#D2945B"]
        heading = 80
        x, y = base_x, base_y
        seg = 12
        length = 30
        for i in range(seg):
            col = tail_colors[i % 2]
            t.pencolor(col)
            move(x, y)
            t.setheading(heading + (i % 2) * 25)
            # draw a small curve by using circle with small radius and extent
            r = 30 + i * 3
            extent = 40
            if i % 2 == 0:
                t.circle(-r, extent)
            else:
                t.circle(r, extent)
            # update x,y to tail tip approx (move forward along heading)
            heading += (-1) ** i * 25
            x += math.cos(math.radians(heading)) * (length + i * 1.5)
            y += math.sin(math.radians(heading)) * (length + i * 1.5)

        # tip
        circle_center(x, y, 12, fillcolor="#F6D6A8", pencolor="#8C5A30", pensz=2)

    def draw_collar():
        # collar as half-ring under chin
        t.pensize(6)
        t.pencolor("#D32F2F")
        move(-55, 60)
        t.setheading(0)
        t.circle(55, 120)
        # bell
        circle_center(0, 38, 12, fillcolor="#FFD54F", pencolor="#B58210", pensz=2)
        # clapper
        circle_center(0, 30, 3, fillcolor="#B58210", pencolor="#B58210")
        # highlight on bell
        circle_center(6, 46, 3, fillcolor="#FFF9C4", pencolor="#FFF9C4")

    # Start drawing in layers
    draw_background()
    draw_body()
    draw_tail()
    draw_head()
    draw_eyes_and_nose()
    draw_whiskers()
    draw_paws()
    draw_collar()

if __name__ == "__main__":
    t = Turtle()
    t.speed(0)
    screen = t.getscreen()
    screen.tracer(0)

    draw_image(t)

    t.hideturtle()
    screen.update()
    canvas = screen.getcanvas()
    canvas.postscript(file="img.eps")
    img = Image.open("img.eps")
    img.save("img.png")