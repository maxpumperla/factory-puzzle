"""In which environment are things happening? (where)
The environment specifies what agents can observe and how
they are rewarded for actions."""
from factory.models import Table
from factory.simulation import Factory
from factory.controls import Action, do_action
from factory.util import print_factory, factory_string
from factory.util.features import has_core_neighbour, get_neighbour_observations, get_one_hot_observations
from factory.config import SIMULATION_CONFIG, factory_from_config

from copy import deepcopy

import gym
from gym import spaces
from ray import rllib
import numpy as np
from typing import Dict


def get_observations(agent_id: int, factory: Factory) -> np.ndarray:
    """Get observation of one agent given the current factory state.
    """
    return get_one_hot_observations(agent_id, factory)


def get_reward(agent_id: int, factory: Factory) -> float:
    """Get the reward for a single agent in its current state.
    """
    moves = factory.moves
    max_num_steps = factory.max_num_steps
    steps = factory.step_count

    agent: Table = factory.tables[agent_id]
    reward = 0.0

    # sum negative rewards due to collisions and illegal moves
    reward += sum(m.reward() / 100. for m in moves.get(agent_id))

    # high incentive for reaching a target
    time_taken = max(0, (max_num_steps - steps) / float(max_num_steps))
    if agent.is_at_target:
        reward += 30.0 * (1 - time_taken)

    # If an agent without core is close to one with core, let it shy away
    if not agent.has_core():
        reward -= has_core_neighbour(agent.node, factory)

    return reward


def get_done(agent_id: int, factory: Factory) -> bool:
    """We're done with the table if it doesn't have a core anymore or we're out of moves.
    """
    if factory.step_count > factory.max_num_steps:
        return True
    agent: Table = factory.tables[agent_id]
    return not agent.has_core()


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
        self.num_actions = config.get("actions")

        self.current_agent = 0
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Box(
            low=config.get("low"),
            high=config.get("high"),
            shape=(config.get("observations"),),
            dtype=np.float32
        )

    def _step(self, action):
        assert action in range(self.num_actions)

        table = self.factory.tables[self.current_agent]
        action_result = do_action(table, self.factory, Action(action))
        self.factory.add_move(self.current_agent, action_result)

        observations: np.ndarray = get_observations(self.current_agent, self.factory)
        rewards = get_reward(self.current_agent, self.factory)
        done = get_done(self.current_agent, self.factory)
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
        return get_observations(self.current_agent, self.factory)


class RoundRobinFactoryEnv(FactoryEnv):

    def __init__(self, config=None):
        super().__init__(config)
        self.current_agent = 0

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
        self.observation_space = gym.spaces.Box(
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
            self.tracker.add_move(current_agent, action_result)

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
        self.tracker.reset()
        if self.config.get("random_init"):
            self.factory = factory_from_config(self.config)
        else:
            self.factory = deepcopy(self.initial_factory)
        return {i: get_observations(i, self.factory) for i in range(self.num_agents)}
