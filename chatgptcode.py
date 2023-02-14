import math
import sys
from typing import List
from typing import Tuple

EPSILON = sys.float_info.epsilon
Point = Tuple[int, int]

def base_case_hull(points: List[Point]) -> List[Point]:
    """ Base case of the recursive algorithm.
    """
    # Return the points as is if there are 1 or 2 points
    if len(points) <= 2:
        return points

    # Compute the line equation of the segment between the first and last points
    # This line separates the points into two groups: above and below the line
    p1, p2 = points[0], points[-1]
    above, below = [], []
    for p in points[1:-1]:
        if is_clockwise(p1, p, p2):
            above.append(p)
        else:
            below.append(p)

    # Compute the upper and lower hulls recursively
    upper = base_case_hull(above)
    lower = base_case_hull(below)

    # Concatenate the hulls and return the result
    return upper + lower

def compute_hull(points: List[Point]) -> List[Point]:
    """
    Given a list of points, computes the convex hull around those points
    and returns only the points that are on the hull.
    """
    # Compute the convex hull using the divide-and-conquer algorithm
    n = len(points)
    if n <= 2:
        return points
    elif n == 3:
        if is_counter_clockwise(*points):
            return points
        else:
            return [points[0], points[2]]
    else:
        # Divide the points into two groups
        mid = n // 2
        left = points[:mid]
        right = points[mid:]

        # Recursively compute the convex hull for each group
        left_hull = compute_hull(left)
        right_hull = compute_hull(right)

        # Merge the two hulls into a single convex hull
        return merge_hulls(left_hull, right_hull)


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

def is_clockwise(a: Point, b: Point, c: Point) -> bool:
    """
    Given three points a,b,c,
    returns True if and only if a,b,c represents a clockwise sequence
    (subject to floating-point precision)
    """
    ax, ay = a
    bx, by = b
    cx, cy = c
    return ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) > 0
