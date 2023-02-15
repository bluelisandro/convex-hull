import math
import sys
from typing import List
from typing import Tuple

EPSILON = sys.float_info.epsilon
Point = Tuple[int, int]

# https://algorithmtutor.com/Computational-Geometry/An-efficient-way-of-merging-two-convex-hull/
# https://algorithmtutor.com/Computational-Geometry/Convex-Hull-Algorithms-Divide-and-Conquer/


def y_intercept(p1: Point, p2: Point, x: int) -> float:
    """
    Given two points, p1 and p2, an x coordinate from a vertical line,
    compute and return the the y-intercept of the line segment p1->p2
    with the vertical line passing through x.
    """
    x1, y1 = p1
    x2, y2 = p2
    slope = (y2 - y1) / (x2 - x1)
    return y1 + (x - x1) * slope


def triangle_area(a: Point, b: Point, c: Point) -> float:
    """
    Given three points a,b,c,
    computes and returns the area defined by the triangle a,b,c.
    Note that this area will be negative if a,b,c represents a clockwise sequence,
    positive if it is counter-clockwise,
    and zero if the points are collinear.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c
    return ((cx - bx) * (by - ay) - (bx - ax) * (cy - by)) / 2


def is_clockwise(a: Point, b: Point, c: Point) -> bool:
    """
    Given three points a,b,c,
    returns True if and only if a,b,c represents a clockwise sequence
    (subject to floating-point precision)
    """
    return triangle_area(a, b, c) < -EPSILON


def is_counter_clockwise(a: Point, b: Point, c: Point) -> bool:
    """
    Given three points a,b,c,
    returns True if and only if a,b,c represents a counter-clockwise sequence
    (subject to floating-point precision)
    """
    # Given three points a,b,c, returns True if and only if a,b,c represents a counter-clockwise sequence (subject to floating-point precision)

    return triangle_area(a, b, c) > EPSILON


def collinear(a: Point, b: Point, c: Point) -> bool:
    """
    Given three points a,b,c,
    returns True if and only if a,b,c are collinear
    (subject to floating-point precision)
    """
    return abs(triangle_area(a, b, c)) <= EPSILON


def clockwise_sort(points: List[Point]):
    """
    Given a list of points, sorts those points in clockwise order about their centroid.
    Note: this function modifies its argument.
    """
    # get mean x coord, mean y coord
    x_mean = sum(p[0] for p in points) / len(points)
    y_mean = sum(p[1] for p in points) / len(points)

    def angle(point: Point):
        return (math.atan2(point[1] - y_mean, point[0] - x_mean) + 2 * math.pi) % (2 * math.pi)

    points.sort(key=angle)
    return


def base_case_hull(points: List[Point]) -> List[Point]:
    """ Base case of the recursive algorithm.
    """
    # TODO: You need to implement this function.

    return points

def merge_hulls(left: List[Point], right: List[Point]) -> List[Point]:
    """Merges the convex hulls of the left and right groups of points
    """
    # Find the rightmost point of the left hull and the leftmost point of the right hull
    leftmost = max(left, key=lambda p: p[0])
    rightmost = min(right, key=lambda p: p[0])

    # Compute the lower tangent of the two hulls
    while True:
        i = left.index(leftmost)
        j = right.index(rightmost)

        a = left[(i - 1) % len(left)]
        b = left[i]
        c = right[(j + 1) % len(right)]
        if is_clockwise(a, b, c):
            leftmost = a
        else:
            break

    # Compute the upper tangent of the two hulls
    while True:
        i = right.index(rightmost)
        j = left.index(leftmost)

        a = right[(i - 1) % len(right)]
        b = right[i]
        c = left[(j + 1) % len(left)]
        if is_clockwise(a, b, c):
            rightmost = a
        else:
            break

    # Concatenate the two hulls and remove any duplicate points
    result = left + right
    result = sorted(set(result), key=lambda p: p[0])

    # Return the points that form the convex hull
    return base_case_hull(result)


def compute_hull(points: List[Point]) -> List[Point]:
    """
    Given a list of points, computes the convex hull around those points
    and returns only the points that are on the hull.
    """
    # TODO: Implement a correct computation of the convex hull
    #  using the divide-and-conquer algorithm.
    
    # TODO: Document your Initialization, Maintenance and Termination invariants.

    # Sort points by X coordinate
    points.sort(key=lambda p: p[0])

    # Divide points into left and right halves to create vertical line
    # Get median of points
    median = len(points) // 2 # this probably should be two lists split in the middle, and passed in recursively
    left_hull = points[0:median]
    right_hull = points[median:]

    # Recursively compute hulls for left and right halves
    compute_hull(left_hull)
    compute_hull(right_hull)

    # Merge convex hulls
    

    return points