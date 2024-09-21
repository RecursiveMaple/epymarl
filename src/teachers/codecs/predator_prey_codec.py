import numpy as np
from gymnasium.spaces import unflatten

from .actors import REGISTRY as actor_registry

INTRO = """
You are a commander controlling two workers to dismantle mines in a vast sea as quickly as possible. Each mine requires two people to dismantle together. Each worker can perform the following actions: explore, goto mine X, stay put. Please briefly inference what the workers should do in current state, then specify next action for each worker. Refer to the example below and output your answer in the format of the yaml file.
Example:
My prompt: Workers can see each other. There are 3 mines in their view: (m0,m1,m2). The distance of each mine relative to worker0: (m0:near,m1:medium,m2:near), relative to worker1: (m0:near,m1:far,m2:near).
Your answer: (less than 50 words)
inference: Since m0 is close to both workers, they should work on m0 together.
actions:
  agent0: [goto mine m0]
  agent1: [goto mine m0]
(no output after this)
"""
TEMPLATE0 = "Workers can not see each other. Worker0 sees {} mines: ({}), with relative distance: ({}). Worker1 sees {} mines: ({}), with relative distance: ({})."
TEMPLATE1 = "Workers can see each other. There are {} mines in their view: ({}). The distance of each mine relative to worker0: ({}), relative to worker1: ({})."
PLAN_SPACE = [
    "explore",
    "goto mine",
    "dismantle mine",
    "stay put",
]


class PredatorPreyCodec:
    def __init__(self, env_args, env):
        self.args = env_args
        self.obs_space = env._posg_to_gym(env._env.unwrapped.observation_spaces)
        self.obs_dim = env_args.get("obs_dim", 2)
        self.obs_size = self.obs_dim * 2 + 1

        self.common_text = INTRO
        self.plan_space = PLAN_SPACE
        self._mines_dicts = []

    def reset(self):
        pass

    def encode(self, obs: tuple) -> str:
        """
        obs: tuple of unwrapped observations returned by self.get_obs()
        fill TEMPLATE0 or TEMPLATE1 with the given obs
        set self._mines_dicts: [{"s0":((x, y),d), ...},{...}]
        """
        self._mines_dicts.clear()
        format_params = []
        if len(obs) == 2:
            total_m = 0
            for mines in obs:
                mines_dict = {}
                mines = [((x, y), abs(x) + abs(y) - 1) for x, y in mines]
                format_params.append(len(mines))
                for mine in sorted(mines, key=lambda x: x[1]):
                    mines_dict["m" + str(total_m)] = mine
                    total_m += 1
                format_params.append(",".join(mines_dict.keys()))
                format_params.append(
                    ",".join(
                        f"{k}:{'far' if v[1]>3 else 'near' if v[1]<2 else 'medium'}"
                        for k, v in mines_dict.items()
                    )
                )
                self._mines_dicts.append(mines_dict)
            return TEMPLATE0.format(*format_params)
        else:
            mines0, mines1, _ = obs
            mines = []
            mines_dict = {}
            for i in range(len(mines0)):
                x0, y0 = mines0[i]
                x1, y1 = mines1[i]
                mines.append(
                    ((x0, y0), abs(x0) + abs(y0) - 1), ((x1, y1), abs(x1) + abs(y1) - 1)
                )
            ids = []
            for i, mine in enumerate(sorted(mines, key=lambda x: x[0][1])):
                id = "m" + str(i)
                mines_dict[id] = mine
                ids.append(id)
            format_params.extend([len(ids), ",".join(ids)])
            format_params.append(
                ",".join(
                    f"{k}:{'far' if v[0][1]>3 else 'near' if v[0][1]<2 else 'medium'}"
                    for k, v in mines_dict.items()
                )
            )
            format_params.append(
                ",".join(
                    f"{k}:{'far' if v[1][1] > 3 else 'near' if v[1][1] < 2 else 'medium'}"
                    for k, v in mines_dict.items()
                )
            )
            self._mines_dicts.append({k: v[0] for k, v in mines_dict.items()})
            self._mines_dicts.append({k: v[1] for k, v in mines_dict.items()})
            return TEMPLATE1.format(*format_params)

    def decode(self, obs: list, description: str, plans: dict[tuple[str], float]):
        """
        obs: same as in self.encode()
        description: the encoded description of the current observation
        plans: map from agents plan(tuple) to probability(float)
        """
        actions = [[0] * 5] * 2
        mines0, mines1 = self._mines_dicts
        for plan, prob in plans.items():
            plan0, plan1 = plan
            if plan0.startswith("dismantle"):
                plan0 = "goto" + plan0[9:]
            if plan1.startswith("dismantle"):
                plan1 = "goto" + plan1[9:]
            action0 = actor_registry[self.actor_mapping(plan0)](
                *self.param_mapping(mines0, description, plan0)
            )
            action1 = actor_registry[self.actor_mapping(plan1)](
                *self.param_mapping(mines1, description, plan1)
            )
            actions[0] = [p + dp * prob for p, dp in zip(actions[0], action0)]
            actions[1] = [p + dp * prob for p, dp in zip(actions[1], action1)]

        return actions

    def param_mapping(self, obs: list, description: str, plan: str) -> tuple:
        """
        obs: mines_dict at local coordinate system of agentX
        plan: the plan of the agentX
        """
        if plan.startswith("explore"):
            return tuple()
        elif plan.startswith("goto"):
            if not obs:
                return ((0, 0), (0, 0), tuple())
            mine = plan.rsplit(" ", 1)[1]
            if mine in obs:
                grid1 = obs[mine][0]
            else:
                grid1 = next(iter(obs.values()))[0]
            x1, y1 = grid1
            walls = [
                ((x, y), d)
                for (x, y), d in obs.values()
                if 0 <= x <= x1 and 0 <= y <= y1 and (x, y) != grid1
            ]
            walls.sort(key=lambda x: (x[1], abs(x[0][0])))
            walls = tuple(w[0] for w in walls)
            return ((0, 0), grid1, walls)
        else:
            return tuple()

    def actor_mapping(self, plan: str) -> str:
        if plan.startswith("explore"):
            return "random"
        elif plan.startswith("goto"):
            return "goto_withwall"
        else:
            return "stay"

    def get_obs(self, obs: list[np.ndarray]) -> tuple[tuple]:
        """
        assume n_agents = 2
        if agents can see each other, return (mines0, mines1, (dx, dy))
        else return (mines0, mines1)
        """
        obs = [unflatten(s, o) for o, s in zip(obs, self.obs_space)]
        obs = [[int(v) for v in o] for o in obs]
        obs[0][int(self.obs_size * self.obs_size / 2)] = 1
        other = None
        if 2 in obs[0]:
            other = divmod(obs[0].index(2), self.obs_size)
            other = (other[1] - self.obs_dim, other[0] - self.obs_dim)
        mines0 = [divmod(i, self.obs_size) for i, v in enumerate(obs[0]) if v == 3]
        mines0 = [(c - self.obs_dim, r - self.obs_dim) for r, c in mines0]
        mines1 = [divmod(i, self.obs_size) for i, v in enumerate(obs[1]) if v == 3]
        mines1 = [(c - self.obs_dim, r - self.obs_dim) for r, c in mines1]

        if other is None:
            return mines0, mines1
        dx, dy = other
        mines0_ = list(set([(x + dx, y + dy) for x, y in mines1] + mines0))
        mines1_ = [(x - dx, y - dy) for x, y in mines0_]
        return mines0_, mines1_, other
