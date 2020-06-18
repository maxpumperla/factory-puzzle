"""In which environment are things happening? (where)
The environment specifies what agents can observe and how
they are rewarded for actions."""
from factory.controls import Action, do_action
from factory.models import Direction
from factory.util import print_factory, factory_string
from factory.features import *
from factory.config import SIMULATION_CONFIG, factory_from_config, MASK_KEY, OBS_KEY

import importlib
from copy import deepcopy

import gym
from gym import spaces
import ray
from ray import rllib
import numpy as np
from typing import Dict


__all__ = ["FactoryEnv", "RoundRobinFactoryEnv", "MultiAgentFactoryEnv", "register_env_from_config",
           "get_observation_space"]


def register_env_from_config():
    env = SIMULATION_CONFIG.get("env")
    cls = getattr(importlib.import_module('factory.environments'), env)
    ray.tune.registry.register_env("factory", lambda _: cls())


def add_masking(self, observations):
    """Add masking, if configured, otherwise return observations as they were."""
    if self.masking:
        if self.config.get("env") == "MultiAgentFactoryEnv":  # Multi-agent scenario
            for key, obs in observations.items():
                observations[key] = {
                    MASK_KEY: update_action_mask(self, agent=key),
                    OBS_KEY: obs,
                }
        else:
            observations = {
                MASK_KEY: update_action_mask(self),
                OBS_KEY: observations,
            }
    return observations


def update_action_mask(env, agent=None):
    current_agent = agent if agent else env.current_agent
    agent_node = env.factory.tables[current_agent].node

    return np.array([
        agent_node.has_neighbour(Direction.up),
        agent_node.has_neighbour(Direction.right),
        agent_node.has_neighbour(Direction.down),
        agent_node.has_neighbour(Direction.left),
        0.0,
    ])


def get_observation_space(config, factory=None) -> spaces.Space:
    if not factory:
        factory = factory_from_config(config)
    dummy_obs = get_observations(0, factory)

    masking = config.get("masking")
    num_actions = config.get("actions")

    observation_space = spaces.Box(
        low=config.get("low"),
        high=config.get("high"),
        shape=(len(dummy_obs),),
        dtype=np.float32
    )

    if masking:  # add masking
        observation_space = spaces.Dict({
            MASK_KEY: spaces.Box(0, 1, shape=(num_actions,)),
            OBS_KEY: observation_space,

        })
    return observation_space


def get_action_space(config):
    return spaces.Discrete(config.get("actions"))


class FactoryEnv(gym.Env):
    """Define a simple OpenAI Gym environment for a single agent."""

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, config=None):
        if config is None:
            config = SIMULATION_CONFIG
        self.config = config
        self.factory = factory_from_config(config)
        self.initial_factory = deepcopy(self.factory)
        self.num_agents = self.config.get("num_tables")
        self.num_actions = self.config.get("actions")
        self.masking = self.config.get("masking")
        self.action_mask = None
        self.current_agent = 0

        self.action_space = get_action_space(self.config)
        self.observation_space = get_observation_space(self.config, self.factory)

    def _step(self, action):
        assert action in range(self.num_actions)

        table = self.factory.tables[self.current_agent]
        action_result = do_action(table, self.factory, Action(action))
        self.factory.add_move(self.current_agent, Action(action), action_result)

        observations: np.ndarray = get_observations(self.current_agent, self.factory)
        rewards = get_reward(self.current_agent, self.factory)
        done = get_done(self.current_agent, self.factory)
        if done:
            self.factory.add_completed_step_count()
            self.factory.print_stats()

        observations = add_masking(self, observations)

        return observations, rewards, done, {}

    def step(self, action):
        return self._step(action)

    def render(self, mode='human'):
        if mode == 'ansi':
            return factory_string(self.factory)
        elif mode == 'human':
            return print_factory(self.factory)
        else:
            super(self.__class__, self).render(mode=mode)

    def reset(self):
        if self.config.get("random_init"):
            self.factory = factory_from_config(self.config)
        else:
            self.factory = deepcopy(self.initial_factory)
        observations = get_observations(self.current_agent, self.factory)
        observations = add_masking(self, observations)
        return observations


class RoundRobinFactoryEnv(FactoryEnv):

    def __init__(self, config=None):
        super().__init__(config)

    def step(self, action):
        result = self._step(action)
        # TODO: we get a ton of invalids, something's off
        self.current_agent = (self.current_agent + 1) % self.num_agents
        return result

    def reset(self):
        # TODO refactor
        self.current_agent = 0
        if self.config.get("random_init"):
            self.factory = factory_from_config(self.config)
        else:
            self.factory = deepcopy(self.initial_factory)
        observations = get_observations(self.current_agent, self.factory)
        observations = add_masking(self, observations)
        return observations


class MultiAgentFactoryEnv(rllib.env.MultiAgentEnv, FactoryEnv):
    """Define a ray multi agent env"""

    def __init__(self, config=None):
        super().__init__(config)

    def step(self, action: Dict):
        # TODO: len(action) is mostly < num_agents
        tables = self.factory.tables
        keys = action.keys()
        for current_agent in keys:
            action_id = action.get(current_agent)
            action_result = do_action(tables[current_agent], self.factory, Action(action_id))
            self.factory.add_move(current_agent, Action(action_id), action_result)

        observations = {i: get_observations(i, self.factory) for i in keys}
        rewards = {i: get_reward(i, self.factory) for i in keys}
        dones = {i: get_done(i, self.factory) for i in keys}
        all_done = all(v for k, v in dones.items())
        dones['__all__'] = all_done
        if all_done:
            self.factory.add_completed_step_count()
            self.factory.print_stats()

        observations = add_masking(self, observations)
        return observations, rewards, dones, {}

    def render(self, mode='human'):
        if mode == 'ansi':
            return factory_string(self.factory)
        elif mode == 'human':
            return print_factory(self.factory)
        else:
            super(self.__class__, self).render(mode=mode)

    def reset(self):
        if self.config.get("random_init"):
            self.factory = factory_from_config(self.config)
        else:
            self.factory = deepcopy(self.initial_factory)
        observations = {i: get_observations(i, self.factory) for i in range(self.num_agents)}
        observations = add_masking(self, observations)
        return observations
