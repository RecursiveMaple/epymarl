from . import actor_funcs as af

REGISTRY = {}

REGISTRY["explore"] = af.grid0_explore_grids_no_wall
REGISTRY["explore_withwall"] = af.grid0_explore_grids_with_wall
REGISTRY["goto"] = af.grid0_goto_grid1_no_wall
REGISTRY["goto_withwall"] = af.grid0_goto_grid1_with_wall
REGISTRY["random"] = af.random_walk
REGISTRY["stay"] = af.stand_still
