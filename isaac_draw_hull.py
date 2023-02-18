from colorsys import hsv_to_rgb
import copy
from tkinter import Button
from tkinter import Canvas
from tkinter import Tk

from convex_hull import compute_hull


def draw_point(canvas, x, y):
    RADIUS = 4
    canvas.create_oval(
        x - RADIUS, y - RADIUS,
        x + RADIUS, y + RADIUS,
        fill="#FFFFFF",
        outline="",
        tags="point",
    )


def add_point(x, y):
    # Don't add points that already exist
    if (x, y) in points:
        return

    draw_point(w, x, y)
    points.add((x, y))

    # Update the hull
    if len(points) > 1:
        draw_hull()


def add_point_click(event):
    add_point(event.x, event.y)


def add_point_grid(event):
    # Snap points to a multiple of `grid` pixels (for debugging)
    grid = 50
    x = round(event.x / grid) * grid
    y = round(event.y / grid) * grid
    add_point(x, y)


def reset(_):
    points.clear()
    w.delete("hull", "point")


def draw_hull():
    hull = copy.copy(compute_hull(list(points)))
    hull.append(hull[0])

    def point_color(frac: float) -> str:
        val = round((1 - frac) * 255)
        return f"#{val:02X}{val:02X}{val:02X}"

    def line_color(frac: float) -> str:
        (r, g, b) = hsv_to_rgb(frac, 1.0, 1.0)
        return f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"

    # Remove the old hull
    w.delete("hull")

    # Draw the lines
    for i in range(0, len(hull) - 1):
        x1 = hull[i][0]
        y1 = hull[i][1]
        x2 = hull[i + 1][0]
        y2 = hull[i + 1][1]
        w.create_line(
            x1, y1, x2, y2,
            width=2,
            fill=line_color(i / (len(hull) - 1)),
            tags=["hull"],
        )

    # Draw the hull points
    for i in range(0, len(hull) - 1):
        RADIUS = 6
        w.create_oval(
            hull[i][0] - RADIUS, hull[i][1] - RADIUS,
            hull[i][0] + RADIUS, hull[i][1] + RADIUS,
            fill=point_color(i / (len(hull) - 2)),
            outline="#000000",
            width=2,
            tags=["hull"],
        )

    return


if __name__ == '__main__':
    master = Tk()
    master.configure(
        background="#202020",
        borderwidth=0,
    )
    points = set()

    canvas_width = 1000
    canvas_height = 800
    w = Canvas(master,
               width=canvas_width,
               height=canvas_height,
               background="#202020",
               highlightthickness=0,
               )
    w.pack()
    w.bind('<Shift-Button-1>', add_point_grid)
    w.bind('<Button-1>', add_point_click)
    master.bind('<space>', reset)
    master.bind('<Escape>', lambda _: master.destroy())

    w.mainloop()
