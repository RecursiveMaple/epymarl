import math
from collections import deque
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=100)
def count_paths(m: int, n: int) -> int:
    """improve with runtime m >= n"""
    return math.comb(m + n, n)


@lru_cache(maxsize=100)
def count_paths_after_first_step(dx: int, dy: int) -> tuple[int]:
    """improve with runtime dx >= dy"""
    if dx == 0 and dy == 0:
        return (0, 0)
    elif dx == 0:
        return (0, 1)
    elif dy == 0:
        return (1, 0)
    first_step_x = count_paths(dx - 1, dy) if dx - 1 > dy else count_paths(dy, dx - 1)
    first_step_y = count_paths(dx, dy - 1)
    return (first_step_x, first_step_y)


@lru_cache(maxsize=100)
def count_paths_with_wall_N(m: int, n: int, walls: tuple[tuple[int]]) -> int:
    """improve with runtime m >= n and walls are sorted small to large"""
    paths = {(0, 0): 1}
    for x, y in walls:
        paths[(x, y)] = 0
    for x in range(m + 1):
        for y in range(n + 1):
            if (x, y) in paths:
                continue
            paths[(x, y)] = paths.get((x - 1, y), 0) + paths.get((x, y - 1), 0)
    return paths[(m, n)]


@lru_cache(maxsize=100)
def count_paths_with_wall_M(m: int, n: int, walls: tuple[tuple[int]]) -> int:
    """improve with runtime m >= n and walls are sorted small to large"""
    paths = {}
    for x, y in walls:
        paths[(x, y)] = count_paths(x, y) if x > y else count_paths(y, x)
    result = count_paths(m, n)
    for i, ((x0, y0), dp) in enumerate(paths.items()):
        for x1, y1 in list(paths.keys())[i:]:
            if x1 >= x0 and y1 >= y0:
                dx, dy = x1 - x0, y1 - y0
                paths[(x1, y1)] -= dp * (
                    count_paths(dx, dy) if dx > dy else count_paths(dy, dx)
                )
        dx, dy = m - x0, n - y0
        result -= dp * (count_paths(dx, dy) if dx > dy else count_paths(dy, dx))
    return result


# TODO: perform A* when these two directions are fully blocked
@lru_cache(maxsize=100)
def count_paths_after_first_step_with_wall(
    dx: int, dy: int, walls: tuple[tuple[int]]
) -> tuple[int]:
    """improve with runtime dx >= dy, walls are sorted small to large"""
    if dx == 0 and dy == 0:
        return (0, 0)
    count_algo = count_paths_with_wall_M
    # if (dx + 1) * (dy + 1) < len(walls) ** 2 / 2:
    #     count_paths_algo = count_paths_with_wall_N
    if count_algo(dx, dy, walls) == 0:
        return (0, 0)
    elif dx == 0:
        return (0, 1)
    elif dy == 0:
        return (1, 0)
    first_x_walls = (
        tuple([(x - 1, y) for x, y in walls if x > 0])
        if dx - 1 >= dy
        else tuple([(y, x - 1) for x, y in walls if x > 0])
    )
    first_step_x = (
        count_algo(dx - 1, dy, first_x_walls)
        if dx - 1 >= dy
        else count_algo(dy, dx - 1, first_x_walls)
    )
    first_y_walls = tuple([(x, y - 1) for x, y in walls if y > 0])
    first_step_y = count_algo(dx, dy - 1, first_y_walls)
    return (first_step_x, first_step_y)


@lru_cache(maxsize=100)
def normalize_list(actions: tuple[float]) -> list[float]:
    """make sure actions are hashable"""
    total = sum(actions)
    return [a / total for a in actions]


# TODO: speed up
def find_nearest_unvisited(
    grid0: tuple[int], unvisited: np.ndarray
) -> tuple[int] | None:
    """BFS, baby!"""
    rows, cols = unvisited.shape
    queue = deque([grid0])
    visited = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.popleft()
        if (x, y) in visited:
            continue
        visited.add((x, y))

        if unvisited[x, y]:
            return (x, y)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                queue.append((nx, ny))
    return None
