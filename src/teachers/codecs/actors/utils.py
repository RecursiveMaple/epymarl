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
    first_step_y = count_paths(dx, dy - 1) if dx > dy - 1 else count_paths(dy - 1, dx)
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
