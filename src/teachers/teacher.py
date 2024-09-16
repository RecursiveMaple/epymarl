from .codecs import REGISTRY as codec_REGISTRY
from .planners import LLMPlanner

# TODO: implement training with llm for every xxx_leaner


class LLMTeacher:
    def __init__(self, args, env, shared_cache=None):
        self.args = args
        self.codec = codec_REGISTRY[args.env_args["key"]](args.env_args, env)
        self.plan_space = self.codec.plan_space
        self.planner = LLMPlanner(args, self.plan_space, shared_cache)

    def select_actions(self, raw_obs, raw_avail_actions=None):
        unwrapped_obs = self.codec.get_obs(raw_obs)
        description = self.codec.encode(unwrapped_obs)
        plans = self.planner.plan(self.codec.common_text, description)
        actions = self.codec.decode(unwrapped_obs, description, plans)
        if raw_avail_actions is not None:
            for agent_idx in range(len(actions)):
                modified = False
                for action_idx in range(len(actions[agent_idx])):
                    if raw_avail_actions[agent_idx][action_idx] == 0:
                        actions[agent_idx][action_idx] = 0
                        modified = True
                if modified:
                    total = sum(actions[agent_idx])
                    if total > 0:
                        actions[agent_idx] = [a / total for a in actions[agent_idx]]
        return actions

    def reset(self):
        self.codec.reset()

    def close(self):
        self.planner.close()
