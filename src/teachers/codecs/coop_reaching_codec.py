import numpy as np
from gymnasium.spaces import unflatten

from .actors import REGISTRY as actor_registry

INTRO = """
On a rectangular 2D grid map, there are two agents whose task is to reach the goal grid at the same time. Agents can only see other agents in a small area around themselves. Agents can not see goal, only when two agents stand on a grid at the same time will the game host say if the grid is a goal or not. Agents can perform the following actions: explore randomly, goto other agent, follow other agent, stand still. Please briefly inference what the agents should do in current state, then select next action for each agent. Refer to the example below and output your answer in the format of the yaml file.
Example:
My prompt: Agents can see each other.
Your answer: (less than 50 words)
inference: the agents should go to the same grid, then search goals together.
actions:
  agent0: [goto other agent]
  agent1: [goto other agent]
(no output after this)
"""
TEMPLATE = ""
PLAN_SPACE = [
    "explore randomly",
    "goto other agent",
    "follow other agent",
    "stand still",
]


class CoopReachingCodec:
    def __init__(self, env_args, env):
        # TODO: get obs_space in a safe way
        self.args = env_args
        self.grids_size = env_args["size"]
        self.obs_space = env._posg_to_gym(env._env.unwrapped.observation_spaces)
        self.obs_distance = env_args.get("obs_distance", 2 * self.grids_size)

        self.common_text = INTRO
        self.plan_space = PLAN_SPACE
        self._fog_grids = None
        self._goal_grids = None
        self.reset()

    def reset(self):
        self._fog_grids = np.ones((self.grids_size, self.grids_size), dtype=bool)
        self._goal_grids = np.ones((self.grids_size, self.grids_size), dtype=bool)

    def update_fog_(self, agent0_self: list[int], agent1_self: list[int]):
        self.clear_grids_with_square_(self._fog_grids, agent0_self, self.obs_distance)
        self.clear_grids_with_square_(self._fog_grids, agent1_self, self.obs_distance)

    def update_goal_(self, agent0_self: list[int], agent1_self: list[int]):
        if agent0_self == agent1_self:
            self._goal_grids[agent0_self[0], agent0_self[1]] = False

    def clear_grids_with_square_(
        self, grids: np.ndarray, center: list[int], radius: int
    ):
        max_x, max_y = grids.shape
        x, y = center
        start_x = max(0, x - radius)
        end_x = min(max_x, x + radius + 1)
        start_y = max(0, y - radius)
        end_y = min(max_y, y + radius + 1)
        grids[start_x:end_x, start_y:end_y] = False

    def encode(self, obs: list) -> str:
        """obs: list of unwrapped observations returned by self.get_obs()"""
        agent0, agent1 = obs
        agent0_self, agent0_other = agent0
        agent1_self, _ = agent1
        if self._fog_grids.any():
            self.update_fog_(agent0_self, agent1_self)
        if self._goal_grids.any():
            self.update_goal_(agent0_self, agent1_self)

        if agent0_other == (self.grids_size, self.grids_size):
            description = "Agents cannot see each other."
        elif agent0_self == agent1_self:
            description = "Agents are at the same grid."
        else:
            description = "Agents can see each other."
        return description

    def decode(self, obs: list, description: str, plans: dict[tuple[str], float]):
        """
        obs: same as in self.encode()
        description: the encoded description of the current observation
        plans: map from agents plan(tuple) to probability(float)
        """
        actions = [[0] * 5] * 2
        for plan, prob in plans.items():
            obs0, obs1 = obs
            plan0, plan1 = plan
            if plan0 == plan1 == "follow other agent":
                plan0 = plan1 = ("stand still", "stand still")
            action0 = actor_registry[self.actor_mapping(plan0)](
                *self.param_mapping(obs0, description, plan0)
            )
            action1 = actor_registry[self.actor_mapping(plan1)](
                *self.param_mapping(obs1, description, plan1)
            )
            if plan0 == "follow other agent":
                action0 = action1
            elif plan1 == "follow other agent":
                action1 = action0
            actions[0] = [p + dp * prob for p, dp in zip(actions[0], action0)]
            actions[1] = [p + dp * prob for p, dp in zip(actions[1], action1)]

        return actions

    def param_mapping(self, obs: list, description: str, plan: str) -> tuple:
        """
        obs: the observation of the agentX
        plan: the plan of the agentX
        """
        grid_self, grid_other = obs
        if plan.startswith("explore"):
            if description == "Agents are at the same grid.":
                return (grid_self, self._goal_grids)
            else:
                return (grid_self, self._fog_grids)
        elif plan.startswith("goto"):
            return (grid_self, grid_other)
        else:
            return tuple()

    def actor_mapping(self, plan: str) -> str:
        if plan.startswith("explore"):
            return "explore"
        elif plan.startswith("goto"):
            return "goto"
        else:
            return "stay"

    def get_obs(self, obs: list[np.ndarray]) -> list[tuple]:
        # obs[i]: flattened observation, shape (21,) = 10 + 11
        obs = [unflatten(s, o) for o, s in zip(obs, self.obs_space)]
        return obs
