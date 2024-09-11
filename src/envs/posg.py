import warnings
from collections.abc import Iterable

import numpy as np
import posggym as gym
from gymnasium.spaces import flatdim
from posggym.wrappers import FlattenObservations, TimeLimit

import envs.pretrained as pretrained  # noqa

from .multiagentenv import MultiAgentEnv


class PosgGymWrapper(MultiAgentEnv):
    def __init__(
        self,
        key,
        time_limit,
        pretrained_wrapper,
        seed,
        common_reward,
        reward_scalarisation,
        **kwargs,
    ):
        self._env = gym.make(f"{key}", **kwargs)
        self._env = TimeLimit(self._env, max_episode_steps=time_limit)
        self._env = FlattenObservations(self._env)

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        self.n_agents = len(self._env.unwrapped.agents)
        self.episode_limit = time_limit
        self._obs = None
        self._info = None

        self.longest_action_space = max(
            self._posg_to_gym(self._env.action_spaces), key=lambda x: x.n
        )
        self.longest_observation_space = max(
            self._posg_to_gym(self._env.observation_spaces), key=lambda x: x.shape[0]
        )

        self._seed = seed
        try:
            self._env.unwrapped.model.seed(self._seed)
        except:
            self._env.reset(seed=self._seed)

        self.common_reward = common_reward
        if self.common_reward:
            if reward_scalarisation == "sum":
                self.reward_agg_fn = lambda rewards: sum(rewards)
            elif reward_scalarisation == "mean":
                self.reward_agg_fn = lambda rewards: sum(rewards) / len(rewards)
            else:
                raise ValueError(
                    f"Invalid reward_scalarisation: {reward_scalarisation} (only support 'sum' or 'mean')"
                )

    def _posg_to_gym(self, dict):
        keys = sorted(dict.keys(), key=int)
        sorted_data = [dict[key] for key in keys]
        return sorted_data

    def _gym_to_posg(self, list):
        data_dict = {}
        for idx, data in enumerate(list):
            data_dict[str(idx)] = data
        return data_dict

    def _pad_observation(self, obs):
        return [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in obs
        ]

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        actions = self._gym_to_posg([int(a) for a in actions])
        obs, reward, done, truncated, _, self._info = self._env.step(actions)
        obs = self._posg_to_gym(obs)
        reward = self._posg_to_gym(reward)
        done = self._posg_to_gym(done)
        truncated = self._posg_to_gym(truncated)
        self._obs = self._pad_observation(obs)

        if self.common_reward and isinstance(reward, Iterable):
            reward = float(self.reward_agg_fn(reward))
        elif not self.common_reward and not isinstance(reward, Iterable):
            warnings.warn(
                "common_reward is False but received scalar reward from the environment, returning reward as is"
            )
        if isinstance(done, Iterable):
            done = all(done)
        if isinstance(truncated, Iterable):
            truncated = all(truncated)
        return self._obs, reward, done, truncated, self._info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self._obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        raise self._obs[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """Returns the shape of the state"""
        if hasattr(self._env.unwrapped, "state_size"):
            return self._env.unwrapped.state_size
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        valid = flatdim(self._env.action_spaces[str(agent_id)]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        obs, info = self._env.reset(seed=seed, options=options)
        self._obs = self._pad_observation(self._posg_to_gym(obs))
        return self._obs, info

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        return self._env.unwrapped.model.seed(seed)

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
