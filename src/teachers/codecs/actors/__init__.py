from .actor_funcs import (
    grid0_explore_grids_no_wall,
    grid0_goto_grid1_no_wall,
    random_walk,
    stand_still,
)

REGISTRY = {}

REGISTRY["explore"] = grid0_explore_grids_no_wall
REGISTRY["goto"] = grid0_goto_grid1_no_wall
REGISTRY["random"] = random_walk
REGISTRY["stay"] = stand_still
