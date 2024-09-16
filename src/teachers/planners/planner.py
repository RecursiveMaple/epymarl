import os
import pickle
import re
from os.path import abspath, dirname

from .llms import REGISTRY as llm_REGISTRY


class LLMPlanner:
    def __init__(self, args, plan_space: list[str], shared_cache=None):
        self.args = args
        self.plan_space = plan_space
        self.cache = PlanCache(args, shared_cache)
        self.llm = llm_REGISTRY[args.llm](**args.llm_args)

    def plan(self, common_text: str, description: str) -> dict[tuple[str], float]:
        if common_text in self.cache and description in self.cache[common_text]:
            return self.cache[common_text][description]

        plans = {}
        nsample = self.args.llm_nsample
        for _ in range(nsample):
            for _ in range(self.args.llm_nretry):
                result = self.llm.query((common_text, description)).lower()
                plan = self.extract_plan(result)
                if not self.validate_plan(plan):
                    continue
                plans[plan] = plans.get(plan, 0) + 1 / nsample
                break
            else:
                self.cache.save()
                raise ValueError(f"Query for plan failed.")

        if common_text not in self.cache:
            self.cache[common_text] = {}
        self.cache[common_text][description] = plans
        return plans

    def extract_plan(self, result: str) -> tuple[str]:
        """get plans from result"""
        plan = []
        # try to match plan with brackets first
        pattern = r"^\s*agent\d+:\s*\[(.*?)\]"
        plan.extend(re.findall(pattern, result, re.MULTILINE | re.IGNORECASE))

        # if no plans were found with brackets, try without brackets
        if not plan:
            pattern = r"^\s*agent\d+:\s*([^\n,.;:()]+)"
            plan.extend(re.findall(pattern, result, re.MULTILINE | re.IGNORECASE))

        # strip any leading/trailing whitespace from the actions
        plan = [p.strip() for p in plan]
        return tuple(plan)

    def validate_plan(self, plan: tuple[str]) -> bool:
        """plan should match n_agents and plan_space"""
        if len(plan) != self.args.n_agents:
            return False
        if self.plan_space is not None:
            for p in plan:
                if p not in self.plan_space:
                    return False
        return True

    def close(self):
        self.llm.close()
        self.cache.save()


class PlanCache:
    def __init__(self, args, shared_cache=None):
        self.args = args
        self.shared = shared_cache is not None
        self.cache = shared_cache if self.shared else {}
        self.cache_path = os.path.join(
            dirname(dirname(dirname(dirname(abspath(__file__))))),
            args.local_results_path,
            f"{args.llm_cache_path}.pkl",
        )
        if os.path.exists(self.cache_path):
            self.load()

    def save(self):
        with open(self.cache_path, "wb") as f:
            if self.shared:
                pickle.dump(dict(self.cache), f)
            else:
                pickle.dump(self.cache, f)

    def load(self):
        self.cache.clear()
        with open(self.cache_path, "rb") as f:
            cache = pickle.load(f)
            if cache is not None:
                self.cache.update(cache)
        return self.cache

    def __getitem__(self, key):
        return self.cache[key]

    def __setitem__(self, key, value):
        self.cache[key] = value

    def __delitem__(self, key):
        del self.cache[key]

    def __iter__(self):
        return iter(self.cache.keys())

    def __len__(self):
        return len(self.cache)
