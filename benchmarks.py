import time
from random import randint
from typing import List
from typing import Set

import matplotlib.pyplot as plt

from convex_hull import Point
from convex_hull import base_case_hull
from convex_hull import compute_hull


def generate_points(
        num_points: int,
        min_x: int = 0,
        max_x: int = 1_000,
        min_y: int = 0,
        max_y: int = 1_000,
) -> List[Point]:
    """ Creates a list of random and unique points for benchmarking the convex_hull algorithm.

    :param num_points: number of unique points to generate.
    :param min_x: minimum x-coordinate for points
    :param max_x: maximum x-coordinate for points
    :param min_y: minimum y-coordinate for points
    :param max_y: maximum y-coordinate for points
    """
    points: Set[Point] = set()
    while len(points) < num_points:
        points.add((randint(min_x, max_x), randint(min_y, max_y)))
    return list(points)


def run_benchmarks():
    # TODO: Generate points randomly, run your convex hull function,
    #  and record the time it takes on inputs of different sizes.
    # TODO: Plot a graph of runtime vs input size. What can you infer from the shape?

    sizes: List[int] = list(range(0, 1_100_000, 100_000))
    dnc_hull_times: List[float] = list()
    naive_hull_times: List[float] = list()
    for n in sizes:
        print(f'n: {n},', end=' ')

        points = generate_points(n)

        start_time = time.time()
        base_case_hull(points)
        time_taken = time.time() - start_time  # time taken (in seconds) for divide-and-conquer

        print(f'dnc_time_taken: {time_taken:.3f}')
        dnc_hull_times.append(time_taken)

        # # base_case_hull
        # start_time = time.time()
        # base_case_hull(points)
        # time_taken = time.time() - start_time  # time taken (in seconds) for naive

        # print(f'naive_time_taken: {time_taken:.3f}')
        # naive_hull_times.append(time_taken)

    plt.scatter(sizes, dnc_hull_times, c='blue')
    plt.plot(sizes, dnc_hull_times, c='blue', label='Main Algorithm Hull Time Taken (s)')

    # plt.scatter(sizes, naive_hull_times, c='red')
    # plt.plot(sizes, naive_hull_times, c='red', label='Base Case Algorithm Hull Time Taken (s)')

    plt.legend()
    plt.xlabel('Number of Points')
    plt.ylabel('Time Taken (s)')
    plt.title('Time Taken vs Input Size')
    plt.savefig('benchmark_plot.png')
    return

if __name__ == '__main__':
    run_benchmarks()
