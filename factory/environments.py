"""In which environment are things happening? (where)
The environment specifies what agents can observe and how
they are rewarded for actions."""
from factory.controls import Action, do_action
from factory.models import Direction
from factory.util import print_factory, factory_string
from factory.features import *
from factory.config import SIMULATION_CONFIG, factory_from_config

from copy import deepcopy

import gym
from gym import spaces
from ray import rllib
import numpy as np
from typing import Dict

__all__ = ["FactoryEnv", "RoundRobinFactoryEnv", "MultiAgentFactoryEnv"]


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

        self.current_agent = 0
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=self.config.get("low"),
            high=self.config.get("high"),
            shape=(self.config.get("observations"),),
            dtype=np.float32
        )

        # Action masking
        self.masking = self.config.get("masking")
        self.action_mask = None
        if self.masking:
            observations = self.observation_space
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.num_actions,)),
                "observations": observations
            })

    def update_action_mask(self):
        agent_node = self.factory.tables[self.current_agent].node

        self.action_mask = np.array([
            agent_node.has_neighbour(Direction.up),
            agent_node.has_neighbour(Direction.right),
            agent_node.has_neighbour(Direction.down),
            agent_node.has_neighbour(Direction.left),
            0.0,
        ])

    def add_masking_to_obs(self, observations):
        """Add masking, if configured, otherwise return observations as they were."""
        if self.masking:
            self.update_action_mask()
            observations = {
                "action_mask": self.action_mask,
                "observations": observations,
            }
        return observations

    def _step(self, action):
        assert action in range(self.num_actions)

        table = self.factory.tables[self.current_agent]
        action_result = do_action(table, self.factory, Action(action))
        self.factory.add_move(self.current_agent, action_result)

        observations: np.ndarray = get_observations(self.current_agent, self.factory)
        rewards = get_reward(self.current_agent, self.factory)
        done = get_done(self.current_agent, self.factory)

        observations = self.add_masking_to_obs(observations)

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
        observations = self.add_masking_to_obs(observations)
        return observations


class RoundRobinFactoryEnv(FactoryEnv):

    def __init__(self, config=None):
        super().__init__(config)

    def step(self, action):
        result = self._step(action)
        self.current_agent = (self.current_agent + 1) % self.num_agents
        return result


class MultiAgentFactoryEnv(rllib.env.MultiAgentEnv, FactoryEnv):
    """Define a ray multi agent env"""

    def __init__(self, config=None):
        super().__init__(config)

        self.num_agents = self.config.get("num_tables")
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=self.config.get("low"),
            high=self.config.get("high"),
            shape=(self.config.get("observations"),),
            dtype=np.float32
        )

    def step(self, action: Dict):
        tables = self.factory.tables
        keys = action.keys()
        for current_agent in keys:
            action_id = action.get(current_agent)
            assert action_id is not None
            action_result = do_action(tables[current_agent], self.factory, Action(action_id))
            self.factory.add_move(current_agent, action_result)

        observations = {i: get_observations(i, self.factory) for i in keys}
        rewards = {i: get_reward(i, self.factory) for i in keys}
        dones = {i: get_done(i, self.factory) for i in keys}
        all_done = all(v for k, v in dones.items())
        dones['__all__'] = all_done

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
        return {i: get_observations(i, self.factory) for i in range(self.num_agents)}
