import math
import sys
from typing import List
from typing import Tuple
import itertools # for combinations

EPSILON = sys.float_info.epsilon
Point = Tuple[int, int]

# Function that checks if list of points are collinear
def collinear(points: List[Point]) -> bool:
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    for x3, y3 in points[2:]:
        if (y3 - y2) * (x2 - x1) != (y2 - y1) * (x3 - x2):
            return False
    return True

def orientation(p, q, r):
    # Returns the orientation of the triplet (p, q, r).
    # Returns 0 if they are collinear, 1 if clockwise, and 2 if counterclockwise.
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # Collinear
    elif val > 0:
        return 1  # Clockwise orientation
    else:
        return 2  # Counterclockwise orientation

def compute_tangent(left_hull, right_hull, leftmost_point, rightmost_point, direction):
    # Compute the tangent line between the leftmost point and the rightmost point of the two hulls
    # in the specified direction (1 for upper tangent, -1 for lower tangent)
    done = False
    while not done:
        done = True
        if direction == 1:
            for i in range(len(left_hull)):
                if orientation(leftmost_point, rightmost_point, left_hull[i]) == -1:
                    leftmost_point = left_hull[i]
                    done = False
                    break
            for i in range(len(right_hull)):
                if orientation(leftmost_point, rightmost_point, right_hull[i]) == 1:
                    rightmost_point = right_hull[i]
                    done = False
                    break
        else:
            for i in range(len(left_hull)):
                if orientation(leftmost_point, rightmost_point, left_hull[i]) == 1:
                    leftmost_point = left_hull[i]
                    done = False
                    break
            for i in range(len(right_hull)):
                if orientation(leftmost_point, rightmost_point, right_hull[i]) == -1:
                    rightmost_point = right_hull[i]
                    done = False
                    break

    return leftmost_point if direction == 1 else rightmost_point

def merge_hulls(left_hull: List[Point], right_hull: List[Point]) -> List[Point]:
    # Merge two convex hulls into one convex hull
    # Find the rightmost point of the left convex hull and the leftmost point of the right convex hull
    leftmost_point = max(left_hull)
    rightmost_point = min(right_hull)

    # Compute the upper and lower tangents
    upper_tangent = compute_tangent(left_hull, right_hull, leftmost_point, rightmost_point, 1)
    lower_tangent = compute_tangent(left_hull, right_hull, leftmost_point, rightmost_point, -1)

    # Remove interior points from both convex hulls
    new_left_hull = left_hull[left_hull.index(lower_tangent):left_hull.index(upper_tangent)+1]
    new_right_hull = right_hull[right_hull.index(upper_tangent):] + right_hull[:right_hull.index(lower_tangent)+1]

    # Concatenate the two new convex hulls in counterclockwise order
    return new_left_hull + new_right_hull


def base_case_hull(points: List[Point]) -> List[Point]:
    # Compute hull using graham scan algorithm
    # Returns the convex hull of a set of points using the Graham scan algorithm.
    n = len(points)

    # Sort points by y-coordinate (and by x-coordinate in case of a tie)
    sorted_points = sorted(points, key=lambda point: (point[1], point[0]))

    # Find the pivot point (the point with the lowest y-coordinate)
    pivot = sorted_points[0]

    # Sort the remaining points by polar angle in counterclockwise order with respect to the pivot
    polar_sorted_points = sorted(sorted_points[1:], key=lambda point: (orientation(pivot, point, (float('inf'), pivot[1])), -point[1], point[0]))

    # Initialize the stack with the pivot and the first point in the sorted list
    stack = [pivot, polar_sorted_points[0]]

    # Iterate over the remaining points, adding them to the stack if they turn left or popping if they turn right
    for i in range(1, n-1):
        while len(stack) > 1 and orientation(stack[-2], stack[-1], polar_sorted_points[i]) != 2:
            stack.pop()
        stack.append(polar_sorted_points[i])

    return stack

def compute_hull(points: List[Point]) -> List[Point]:
    # Sort points by x-coordinate
    points.sort(key=lambda p: p[0])

    if len(points) <= 3:
        return base_case_hull(points)
    else:
        # Divide
        # Split points in half
        mid = len(points) // 2
        left_hull = points[:mid]
        right_hull = points[mid:]

        # Conquer
        # Recursively find convex hulls for left and right
        left_hull = compute_hull(left_hull)
        right_hull = compute_hull(right_hull)
    
    return merge_hulls(left_hull, right_hull)

# Psudeocode
    # 1 Sort the set of points by their x-coordinate. If two points have the same x-coordinate, sort them by their y-coordinate.

    # 2 If there are only three or fewer points in the set, compute the convex hull directly using a known algorithm (e.g., Graham's scan or Jarvis march). This is the base case for the recursion.

    # 3 Divide the set of points into two roughly equal-sized sets, using a vertical line that divides the points in half.

    # 4 Recursively compute the convex hull of each of the two sets. This can be done by repeating steps 1-3 on each of the two sets.

    # 5 Merge the two convex hulls to obtain the final convex hull. This can be accomplished by connecting the rightmost point of the left convex hull to the leftmost point of the right convex hull, and continuing in a counterclockwise direction until the starting point is reached.