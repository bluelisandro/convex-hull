import math
import sys
import numpy
from typing import List
from typing import Tuple
from itertools import combinations

EPSILON = sys.float_info.epsilon
Point = Tuple[int, int]

# Declare a global boolean variable to only sort list of points once
# global_points_sorted = False

##### ----- Helper Functions ----- #####


def collinear(a: Point, b: Point, c: Point) -> bool:
    """
    Given three points a,b,c,
    returns True if and only if a,b,c are collinear
    (subject to floating-point precision)
    """
    return abs(triangle_area(a, b, c)) <= EPSILON


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

# Computes the cross product of 3 points
def cross_product(p1, p2, p3):
    # Convert tuples to NumPy arrays
    p1 = numpy.array([p1[0], p1[1], 0])
    p2 = numpy.array([p2[0], p2[1], 0])
    p3 = numpy.array([p3[0], p3[1], 0])

    # Compute the cross product
    cross_product = numpy.cross(p2 - p1, p3 - p1)

    return cross_product

def compute_y_intercept(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    slope1 = (y2 - y1) / (x2 - x1)
    y_intercept1 = y1 - slope1 * x1

    x3, y3 = line2[0]
    x4, y4 = line2[1]
    slope2 = (y4 - y3) / (x4 - x3)
    y_intercept2 = y3 - slope2 * x3

    if y_intercept1 == y_intercept2:
        return y_intercept1
    else:
        return None


##### ----- Convex Hull Divide and Conquer Algorithm ----- #####

# def compute_hull(points: List[Point]) -> List[Point]:
def compute_hull_library(points: List[Point]) -> List[Point]:
    from scipy.spatial import ConvexHull
    # Define the points for the convex hull
    points = numpy.array(points)

    # Compute the convex hull
    hull = ConvexHull(points)

    # Extract the vertices of the convex hull
    hull_vertices = hull.vertices

    # Convert the vertices to a list of points as tuples
    hull_points = [points[i] for i in hull_vertices]

    return hull_points

def merge_hulls_library(left_hull: List[Point], right_hull: List[Point]) -> List[Point]:
    from scipy.spatial import ConvexHull
    # Define the points for the first convex hull
    points1 = numpy.array(left_hull)

    # Define the points for the second convex hull
    points2 = numpy.array(right_hull)

    # Compute the convex hulls of the two point sets
    hull1 = ConvexHull(points1)
    hull2 = ConvexHull(points2)

    # Merge the two convex hulls
    combined_points = numpy.concatenate((points1, points2))
    combined_hull = ConvexHull(combined_points)

    # Extract the vertices of the convex hull
    hull_vertices = combined_hull.vertices

    # Convert the vertices to a list of points as tuples
    hull_points = [points_list[i] for i in hull_vertices]

    return hull_points

# Given two convex hulls, hull1 and hull2, computes and returns the convex hull
# that contains all points in both hull1 and hull2.
def merge_hulls(left_hull: List[Point], right_hull: List[Point]) -> List[Point]:
    # Tim's Psuedocode
    # store rightmost point of left hull
    rightmost = max(left_hull, key=lambda p: p[0])[0]

    # store leftmost point of right hull
    leftmost = min(right_hull, key=lambda p: p[0])[0]

    # store middle line of both hulls
    dividing_line = [(rightmost, 0), (leftmost, 0)]

    # declare new leftmost var
    curr_leftmost = None
    # declare new rightmost var
    curr_rightmost = None

    # while leftmost != new leftmost AND rightmost != new rightmost:
    while leftmost != curr_leftmost and rightmost != curr_rightmost:
        # new_rightmost = rightmost
        curr_rightmost = rightmost
        # new_leftmost = leftmost
        curr_leftmost = leftmost
        # y_intercept = y intercept of leftmost, rightmost, and divding line
        y_intercept = compute_y_intercept(
            dividing_line, [(leftmost, 0), (rightmost, 0)])
        # while y intercept < y intercept of points[index of leftmost + 1], rightmost, dividing line
        while y_intercept < compute_y_intercept(dividing_line, [(left_hull[left_hull.index(leftmost) + 1], 0), (rightmost, 0)]):
            # leftmost = right_hull[index of leftmost + 1]
            leftmost = right_hull[right_hull.index(leftmost) + 1]
            # y_intercept = y_intercept of leftmost, rightmost, dividing line
            y_intercept = compute_y_intercept(
                [(leftmost, 0), (rightmost, 0)], dividing_line)
        # while y intercept < y intercept of leftmost, points[index of rightmost], dividing line
        while y_intercept < compute_y_intercept([(left_hull[right_hull[right_hull.index(rightmost) - 1]], 0), (rightmost, 0)], dividing_line):
            # rightmost = left[index of rightmost - 1]
            rightmost = left_hull[left_hull.index(rightmost) - 1]
            # y_intercept = y_intercept of leftmost, rightmost, dividing line
            y_intercept = compute_y_intercept(
                [(leftmost, 0), (rightmost, 0)], dividing_line)

    return left_hull[left_hull.index(leftmost):left_hull.index(rightmost) + 1] + right_hull[right_hull.index(rightmost):right_hull.index(leftmost) + 1]

# Computes convex hull of a set of points using brute force
def base_case_hull(points: List[Point]) -> List[Point]:
    lines = combinations(points, 2)

    convex_hull = set()

    for line in lines:
        p1, p2 = line
        for p3 in points:
            if collinear(p1, p2, p3):
                # convex_hull.add(p1)
                # convex_hull.add(p2)
                convex_hull.add(p3)
            else:
                # If cp is NOT 0, then it has a point on one side, therefore line not on hull
                cp = cross_product(p1, p2, p3)
                # print(cp)
                if numpy.any(cp != 0) and p3 in convex_hull:
                    # Not on hull
                    convex_hull.remove(p3)
                else:
                    # On hull
                    # convex_hull.add(p1)
                    # convex_hull.add(p2)
                    convex_hull.add(p3)

    return list(convex_hull)

    # Psuedocode
    """ get the combination of points (this is your line segments)
        for every line segment:
        point1, point2 = x
        for every point:
            if theyre collinear: ignore
            do cross product
            check if the cp is <0, >0 or is 0
        if there was a line segment where there were no points on one side, then that segment is on the hull """

# Given a list of points, recursively computes the convex hull around those points,
# and returns only the points that are on the hull.


# def compute_hull2(points: List[Point]) -> List[Point]:
def compute_hull(points: List[Point]) -> List[Point]:
    # TODO: Implement a correct computation of the convex hull
    #  using the divide-and-conquer algorithm
    # TODO: Document your Initialization, Maintenance and Termination invariants.

    # Sort the set of points by their x-coordinate.
    # If two points have the same x-coordinate, sort them by their y-coordinate.
    points.sort(key=lambda p: (p[0], p[1]))

    # Base Case
    # If there are only 5 or fewer points in the set,
    # compute the convex hull directly using a known algorithm (in our case, brute force)
    if len(points) <= 5:
        convex_hull = base_case_hull(points)

    # Recursive Case
    # Divide the set of points into two roughly equal-sized sets,
    # then recursively compute the convex hull of each of the two sets using compute_hull()
    else:
        # If all points have same x-coordinate, return points as convex hull
        if len(set(p[0] for p in points)) == 1:
            convex_hull = points
        else:
            # Divide
            median = len(points) // 2
            left_hull = compute_hull(points[:median])
            right_hull = compute_hull(points[median:])
            # Conquer
            convex_hull = merge_hulls(left_hull, right_hull)

    return convex_hull

# Convex Hull Divide and Conquer Psudeocode
    # Sort the set of points by their x-coordinate. If two points have the same x-coordinate, sort them by their y-coordinate.

    # Base Case:
    # If there are only 5 or fewer points in the set,
    # compute the convex hull directly using a known algorithm (in our case, brute force)

    # Recursive Case: Divide the set of points into two roughly equal-sized sets,
    # then recursively compute the convex hull of each of the two sets using compute_hull()

    # Merge the two convex hulls to obtain the final convex hull.
    # This can be accomplished by connecting the rightmost point of the left convex hull
    # to the leftmost point of the right convex hull, and continuing in a counterclockwise
    # direction until the starting point is reached.

# ----- Tests -----

# https://www.desmos.com/calculator/mzbbuzc62z
# https://planetcalc.com/8576/?set2d=0%3B1%0A1%3B0%0A1%3B2%0A2%3B1%0A3%3B1%0A4%3B0%0A4%3B2%0A5%3B1

# Test merge_hulls using two simple 4 point hulls
def test_merge_hulls():
    expected_merged_hull = [(0, 1), (1, 0), (1, 2), (4, 0), (4, 2), (5, 1)]

    left_hull = [(0, 1), (1, 0), (1, 2), (2, 1)]

    right_hull = [(3, 1), (4, 0), (4, 2), (5, 1)]

    merged_hull = merge_hulls(left_hull, right_hull)

    assert merged_hull == expected_merged_hull

# Test base case hull using simple 5 points
def test_base_case_hull():
    print("------------------ Test base_case_hull() ------------------")
    expected_hull = [(0, 1), (1, 0), (1, 2), (2, 1)]

    points = [(0,1),(1,0),(1,2),(2,1),(1,1)]

    hull = base_case_hull(points)

    print("hull: ", hull)

    assert expected_hull == hull
    print("------------------      Test passed!     ------------------")

# test_base_case_hull()