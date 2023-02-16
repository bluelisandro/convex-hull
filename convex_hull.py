import math
import sys
from typing import List
from typing import Tuple
import itertools # for combinations

EPSILON = sys.float_info.epsilon
Point = Tuple[int, int]

# Checks if all points in list are collinear
# Lisandro: I think that this is easier than having args of only 3 points, because we can give it a list with any number of points by passing in [a, b, ...]
def collinear(points: List[Point]) -> bool:
    """ Given a list of points, returns True if and only if all of those points are collinear.
    """
    if len(points) <= 2:
        return True
    for i in range(1, len(points) - 1):
        if not collinear([points[i - 1], points[i], points[i + 1]]):
            return False
    return True
# def collinear(a: Point, b: Point, c: Point) -> bool:
#     """
#     Given three points a,b,c,
#     returns True if and only if a,b,c are collinear
#     (subject to floating-point precision)
#     """
#     return abs(triangle_area(a, b, c)) <= EPSILON

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
    return triangle_area(a, b, c) > EPSILON

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

def cross_product(point1: Point, point2: Point) -> float:
    """
    Given two points, computes and returns the cross product of the two points.
    Note that this cross product will be negative if point1, point2 represents a clockwise sequence,
    positive if it is counter-clockwise,
    and zero if the points are collinear.
    """
    x1, y1 = point1
    x2, y2 = point2
    return (x1 * y2) - (x2 * y1)


def base_case_hull(points: List[Point]) -> List[Point]:
    """ Base case of the recursive algorithm.
    """
    
    # Compute combination of points
    # combinations = itertools.combinations(points, 2)

    # Stores list of circular pairs of points as list of tuples
    # Each pair represents a line segment
    # Example: [(1,2), (2,3), (3,1)] --> [ [(1, 2), (2, 3)], [(2, 3), (3, 1)], [(3, 1), (1, 2)] ]
    combinations = [(point1, point2) for point1, point2 in zip(points, points[1:]+points[:1])]
    # TODO Not sure if these sort of combinations are what we need

    for combination in combinations:
        point1 = combination[0]
        point2 = combination[1]
        if collinear([point1, point2]):
            continue
        else:
            # Compute cross product
            cp = cross_product(point1, point2)
            if cp < 0:
                # Point 1 is on the hull
                # Point 2 is not on the hull
                not_on_hull = points.index(point2)
                points.pop(not_on_hull)
            elif cp > 0:
                # Point 1 is not on the hull
                # Point 2 is on the hull
                not_on_hull = points.index(point1)
                points.pop(not_on_hull)
            else:
                # Point 1 and Point 2 are collinear
                continue
    
    """ get the combination of points (this is your line segments)
        for every line segment:
        point1, point2 = x
        for every point:
            if theyre collinear: ignore
            do cross product
            check if the cp is <0, >0 or is 0
        if there was a line segment where there were no points on one side, then that segment is on the hull """

    return points


def compute_hull(points: List[Point]) -> List[Point]:
    """
    Given a list of points, computes the convex hull around those points
    and returns only the points that are on the hull.
    """
    # TODO: Implement a correct computation of the convex hull
    #  using the divide-and-conquer algorithm
    # TODO: Document your Initialization, Maintenance and Termination invariants.

    points.sort(key=lambda p: p[0])

    if len(points) > 5:
        if collinear(points):
            return points
        else:
            # Divide
            median = len(points) // 2
            left = compute_hull(points[:median])
            right = compute_hull(points[median:])
            # Conquer
            return merge_hulls(left, right)

    else:
        hull = base_case_hull(points)
        return hull