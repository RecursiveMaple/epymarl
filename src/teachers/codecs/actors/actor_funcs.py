"""
Assume DO_NOTHING=0, UP=1, DOWN=2, LEFT=3, RIGHT=4 \n
The origin point (0,0) is left top corner \n
actor_funcs here should return a list of probabilities of actions \n
"""

from functools import lru_cache

import numpy as np

from . import utils


@lru_cache(maxsize=100)
def grid0_goto_grid1_no_wall(grid0: tuple[int], grid1: tuple[int]) -> list[float]:
    """assume both grids are valid"""
    if grid0 == grid1:
        return [1, 0, 0, 0, 0]
    x0, y0 = grid0
    x1, y1 = grid1
    dx = x1 - x0
    dy = y1 - y0
    reverse_x = False
    if dx < 0:
        dx = -dx
        reverse_x = True
    reverse_y = False
    if dy < 0:
        dy = -dy
        reverse_y = True
    transposed = False
    if dx < dy:
        dx, dy = dy, dx
        transposed = True
    if dy == 0:
        first_step_x, first_step_y = 1, 0
    else:
        first_step_x, first_step_y = utils.count_paths_after_first_step(dx, dy)
    if transposed:
        first_step_x, first_step_y = first_step_y, first_step_x

    up, down, left, right = 0, 0, 0, 0
    if first_step_x > 0:
        if reverse_x:
            left = first_step_x
        else:
            right = first_step_x
    if first_step_y > 0:
        if reverse_y:
            up = first_step_y
        else:
            down = first_step_y
    actions = (up, down, left, right)
    actions = utils.normalize_list(actions)
    return [0, *actions]


@lru_cache(maxsize=100)
def grid0_goto_grid1_with_wall(
    grid0: tuple[int], grid1: tuple[int], walls: tuple[tuple[int]]
) -> list[float]:
    """assume grid1 is not wall, walls are sorted small to large"""
    if grid0 == grid1:
        return [1, 0, 0, 0, 0]
    x0, y0 = grid0
    x1, y1 = grid1
    dx = x1 - x0
    dy = y1 - y0
    walls = [(x - x0, y - y0) for x, y in walls]
    reverse_x = False
    if dx < 0:
        dx = -dx
        walls = [(-x, y) for x, y in walls]
        reverse_x = True
    reverse_y = False
    if dy < 0:
        dy = -dy
        walls = [(x, -y) for x, y in walls]
        reverse_y = True
    transposed = False
    if dx < dy:
        dx, dy = dy, dx
        transposed = True
    first_step_x, first_step_y = utils.count_paths_after_first_step(dx, dy)
    if first_step_x == 0 and first_step_y == 0:
        return [1, 0, 0, 0, 0]
    if transposed:
        first_step_x, first_step_y = first_step_y, first_step_x

    up, down, left, right = 0, 0, 0, 0
    if (1, 0) in walls and (0, 1) in walls:
        up = first_step_x
        left = first_step_y
    elif (1, 0) in walls:
        up = first_step_x / 2
        down = first_step_y + first_step_x / 2
    elif (0, 1) in walls:
        left = first_step_y / 2
        right = first_step_x + first_step_y / 2
    else:
        right = first_step_x
        down = first_step_y
    if reverse_x:
        left, right = right, left
    if reverse_y:
        up, down = down, up
    actions = (up, down, left, right)
    actions = utils.normalize_list(actions)
    return [0, *actions]


def grid0_explore_grids_no_wall(
    grid0: tuple[int], unexplored: np.ndarray
) -> list[float]:
    """assume grid0 is valid"""
    grid1 = utils.find_nearest_unvisited(grid0, unexplored)
    if grid1 is None:
        return [1, 0, 0, 0, 0]
    return grid0_goto_grid1_no_wall(grid0, grid1)


def grid0_explore_grids_with_wall(
    grid0: tuple[int], unexplored: np.ndarray, walls: tuple[tuple[int]]
) -> list[float]:
    """assume walls are sorted small to large"""
    grid1 = utils.find_nearest_unvisited(grid0, unexplored)
    if grid1 is None:
        return [1, 0, 0, 0, 0]
    return grid0_goto_grid1_with_wall(grid0, grid1, walls)


def random_walk() -> list[float]:
    return [0, 0.25, 0.25, 0.25, 0.25]


def stand_still() -> list[float]:
    return [1, 0, 0, 0, 0]
