import numpy as np
from gymnasium.spaces import unflatten

from .actors import REGISTRY as actor_registry

INTRO = """
You are a commander controlling two workers at different skill level to dismantle mines in a vast sea as quickly as possible. Some complex mines require two people to work together. Each worker can perform the following actions: explore, goto mine X, stay put. Please briefly inference what the workers should do in current state, then specify next action for each worker. Refer to the example below and output your answer in the format of the yaml file.
Example:
My prompt: Workers can see each other. Worker0 is able to dismantle 1 mines: (s1), worker1 is able to do 2: (s0,s1), and 1 mines need cooperation: (c0). The distance of each mine relative to worker0: (s0:near,s1:medium,c0:near), relative to worker1: (s0:near,s1:far,c0:near).
Your answer: (less than 50 words)
inference: Since c0 is close to both workers, they should work on c0 together. But before that, worker1 is near s0 and is able to dismantle it, so he should do s0 first.
actions:
  agent0: [goto mine c0]
  agent1: [goto mine s0]
(no output after this)
"""
TEMPLATE0 = "Workers can not see each other. Worker0 sees {} simple mines(s) and {} complex mines(c) in view: ({}), with relative distance: ({}). Worker1 sees {} mines: ({}), with relative distance: ({})."
TEMPLATE1 = "Workers can see each other. Worker0 is able to dismantle {} mines: ({}), worker1 is able to do {}: ({}), and {} mines need cooperation: ({}). The distance of each mine relative to worker0: ({}), relative to worker1: ({})."
PLAN_SPACE = [
    "explore",
    "goto mine",
    "stay put",
]


class LBForagingCodec:
    def __init__(self, env_args, env):
        self.args = env_args
        self.obs_space = env._posg_to_gym(env._env.unwrapped.observation_spaces)

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
            total_s, total_c = 0, 0
            for level, mines in obs:
                simple_mines = []
                complex_mines = []
                mines_dict = {}
                for x, y, l in mines:
                    if level >= l:
                        simple_mines.append(((x, y), abs(x) + abs(y) - 1))
                    else:
                        complex_mines.append(((x, y), abs(x) + abs(y) - 1))
                format_params.extend([len(simple_mines), len(complex_mines)])
                for mine in sorted(simple_mines, key=lambda x: x[1]):
                    mines_dict["s" + str(total_s)] = mine
                    total_s += 1
                for mine in sorted(complex_mines, key=lambda x: x[1]):
                    mines_dict["c" + str(total_c)] = mine
                    total_c += 1
                format_params.append(",".join(mines_dict.keys()))
                format_params.append(
                    ",".join(
                        f"{k}:{'far' if v[1]>3 else 'near' if v[1]<2 else 'medium'}"
                        for k, v in mines_dict.items()
                    )
                )
                self._mines_dicts.append(mines_dict)
            format_params[4] += format_params.pop(5)
            return TEMPLATE0.format(*format_params)
        else:
            (l0, mines0), (l1, mines1), _ = obs
            simple_mines = []
            complex_mines = []
            mines_dict = {}
            for i in range(len(mines0)):
                x0, y0, l = mines0[i]
                x1, y1, _ = mines1[i]
                mine_info = ((x0, y0), abs(x0) + abs(y0) - 1, l0 >= l), (
                    (x1, y1),
                    abs(x1) + abs(y1) - 1,
                    l1 >= l,
                )
                if l0 >= l or l1 >= l:
                    simple_mines.append(mine_info)
                else:
                    complex_mines.append(mine_info)
            s0_ids = []
            s1_ids = []
            c_ids = []
            for i, mine in enumerate(sorted(simple_mines, key=lambda x: x[0][1])):
                id = "s" + str(i)
                mines_dict[id] = mine
                if mine[0][2]:
                    s0_ids.append(id)
                if mine[1][2]:
                    s1_ids.append(id)
            for i, mine in enumerate(sorted(complex_mines, key=lambda x: x[0][1])):
                id = "c" + str(i)
                mines_dict[id] = mine
                c_ids.append(id)
            format_params.extend(
                [
                    len(s0_ids),
                    ",".join(s0_ids),
                    len(s1_ids),
                    ",".join(s1_ids),
                    len(c_ids),
                    ",".join(c_ids),
                ]
            )
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
            self._mines_dicts.append({k: v[0][:2] for k, v in mines_dict.items()})
            self._mines_dicts.append({k: v[1][:2] for k, v in mines_dict.items()})
            return TEMPLATE1.format(*format_params)

    def decode(self, obs: list, description: str, plans: dict[tuple[str], float]):
        """
        obs: same as in self.encode()
        description: the encoded description of the current observation
        plans: map from agents plan(tuple) to probability(float)
        """
        actions = [[0] * 6] * 2
        mines0, mines1 = self._mines_dicts
        for plan, prob in plans.items():
            plan0, plan1 = plan
            action0 = None
            if plan0.startswith("goto"):
                target = plan0.rsplit(" ", 1)[1]
                if target in mines0 and mines0[target][1] == 0:
                    action0 = [0] * 5 + [1]
            if action0 is None:
                action0 = actor_registry[self.actor_mapping(plan0)](
                    *self.param_mapping(mines0, description, plan0)
                ) + [0]
            if plan1.startswith("goto"):
                target = plan1.rsplit(" ", 1)[1]
                if target in mines1 and mines1[target][1] == 0:
                    action1 = [0] * 5 + [1]
            if action1 is None:
                action1 = actor_registry[self.actor_mapping(plan1)](
                    *self.param_mapping(mines1, description, plan1)
                ) + [0]
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
        if agents can see each other, return ((l0, mines0), (l1, mines1), (dx, dy))
        else return ((l0, mines0), (l1, mines1))
        """
        obs = [unflatten(s, o) for o, s in zip(obs, self.obs_space)]
        obs = [[int(v) for v in o] for o in obs]
        obs = [[tuple(o[i : i + 3]) for i in range(0, len(o), 3)] for o in obs]
        obs = [(o[:2], list(set(o[2:]) - set(((-1, -1, 0),)))) for o in obs]
        agent0, agent1 = obs
        (agent0_self, agent0_other), agent0_mines = agent0
        agent0_mines = [(x - 2, y - 2, l) for x, y, l in agent0_mines]
        (agent1_self, _), agent1_mines = agent1
        agent1_mines = [(x - 2, y - 2, l) for x, y, l in agent1_mines]
        if agent0_other[0] == -1:
            return ((agent0_self[2], agent0_mines), (agent1_self[2], agent1_mines))
        dx, dy = agent0_other[0] - agent0_self[0], agent0_other[1] - agent0_self[1]
        agent0_mines_ = list(
            set([(x + dx, y + dy, l) for x, y, l in agent1_mines] + agent0_mines)
        )
        agent1_mines_ = [(x - dx, y - dy, l) for x, y, l in agent0_mines_]
        return (
            (agent0_self[2], agent0_mines_),
            (agent1_self[2], agent1_mines_),
            (dx, dy),
        )
