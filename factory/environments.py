"""In which environment are things happening? (where)
The environment specifies what agents can observe and how
they are rewarded for actions."""
from factory.models import Factory
from factory.agents import Agent
from factory.util import print_factory, factory_string

from abc import ABC, abstractmethod
from copy import deepcopy

import gym
from gym import spaces
from ray import rllib
import numpy as np


class Observation(ABC):
    """Get all relevant observations for a specified agent,
    given a factory state."""

    factory: Factory

    @abstractmethod
    def get_reward(self, agent: Agent) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_observations(self, agent) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def done(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def info(self) -> str:
        raise NotImplementedError


class FactoryEnv(gym.Env):
    """Define a simple OpenAI Gym environment for a single agent."""

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, factory: Factory, num_actions: int):
        self.initial_factory = deepcopy(factory)
        self.factory = factory

        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = None  # TODO: define this

    def step(self, action):
        """Run one time-step of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def render(self, mode='human'):
        if mode == 'ansi':
            return factory_string(self.factory)
        elif mode == 'human':
            return print_factory(self.factory)
        else:
            super(FactoryEnv, self).render(mode=mode)

    def reset(self):
        self.factory = deepcopy(self.initial_factory)


class FactoryVectorEnv(rllib.env.VectorEnv, ABC):
    pass


class FactoryMultiAgentEnv(rllib.env.MultiAgentEnv, ABC):
    pass
