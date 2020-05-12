"""In which environment are things happening? (where)
The environment specifies what agents can observe and how
they are rewarded for actions."""
from factory.models import Factory, Table, Direction, Node
from factory.controls import TableController, Action
from factory.util import print_factory, factory_string

from copy import deepcopy

import gym
from gym import spaces
from ray import rllib
import numpy as np
from typing import Callable


def check_neighbour(node: Node, direction: Direction, factory: Factory):
    """If an agent has a neighbour in the specified direction, add a 1,
    else 0 to the observation space. If that neighbour is free, add 1,
    else 0 (a non-existing neighbour counts as occupied).
    """
    has_direction = node.has_neighbour(direction)
    is_occupied = True
    if has_direction:
        node: Node = node.get_neighbour(direction)
        if node.is_rail:
            rail = factory.get_rail(node)
            is_occupied = rail.shuttle_node().has_table()
        else:
            is_occupied = node.has_table()
    return [has_direction, not is_occupied]


def get_observations(agent_id: int, factory: Factory) -> np.array:
    """Get observation of one agent (here the same as Table).
    """
    # Agent ID and coordinates (3)
    agent: Table = factory.tables[agent_id]
    obs = [agent_id]
    obs += list(agent.node.coordinates)

    # Direct neighbours available and free? (8)
    obs += check_neighbour(agent.node, Direction.up, factory)
    obs += check_neighbour(agent.node, Direction.right, factory)
    obs += check_neighbour(agent.node, Direction.down, factory)
    obs += check_neighbour(agent.node, Direction.left, factory)

    # Has core and core coordinates? (3)
    obs.append(agent.has_core())
    if agent.has_core():
        current_target: Node = agent.core.current_target
        obs += list(current_target.coordinates)
    else:
        obs += [-1, -1]
    return np.asarray(obs)


def get_observation_space() -> gym.Space:
    return gym.Space.Box(low=-1, high=10, shape=(14,), dtype=np.float32)


def get_reward(agent_id: int, factory: Factory) -> float:
    """Get the reward for a single agent in its current state.

    TODO: count invalid moves (particularly collisions) accumulated in factory
    TODO: account for time/steps taken
    TODO: if table has no core, avoid other targets and agents
    TODO: if table has core, slightly incentivize being close to target, but more so to reach it
    """
    agent: Table = factory.tables[agent_id]
    reward = 0.0
    if agent.has_core():
        reward += agent.core.num_phases - len(agent.core.cycle)
    return reward


def get_done(agent_id: int, factory: Factory) -> bool:
    """We're done with the table if it doesn't have a core anymore.

    TODO: track maximum number of steps
    """
    agent: Table = factory.tables[agent_id]
    return not agent.has_core()


class FactoryEnv(gym.Env):
    """Define a simple OpenAI Gym environment for a single agent."""

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, factory: Factory, observation_fct: Callable, reward_fct: Callable,
                 done_fct: Callable, observation_space: Callable, info_fct: Callable = None):
        self.initial_factory = deepcopy(factory)
        self.factory = factory
        self.action_space = spaces.Discrete(5)  # agents have single discrete actions
        self.observation_space = observation_space()
        self.observation_fct = observation_fct
        self.reward_fct = reward_fct
        self.done_fct = done_fct
        self.info_fct = info_fct

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
        table = self.factory.tables[0]
        controller = TableController(table, self.factory)
        controller.take_action(Action(action))

        observations = self.observation_fct(0, self.factory)
        rewards = self.reward_fct(0, self.factory)
        done = self.done_fct(0, self.factory)
        # TODO info
        info = {}
        return observations, rewards, done, info

    def render(self, mode='human'):
        if mode == 'ansi':
            return factory_string(self.factory)
        elif mode == 'human':
            return print_factory(self.factory)
        else:
            super(FactoryEnv, self).render(mode=mode)

    def reset(self):
        self.factory = deepcopy(self.initial_factory)


class FactoryMultiAgentEnv(rllib.env.MultiAgentEnv):
    """Define a ray multi agent env"""

    def __init__(self, factory: Factory, observation_fct: Callable, reward_fct: Callable,
                 done_fct: Callable, info_fct: Callable = None):
        self.initial_factory = deepcopy(factory)
        self.factory = factory

        self.observation_fct = observation_fct
        self.reward_fct = reward_fct
        self.done_fct = done_fct
        self.info_fct = info_fct

    def step(self, action):
        tables = self.factory.tables
        controllers = [TableController(t, self.factory) for t in tables]
        for i, controller in enumerate(controllers):
            controller.take_action(Action(action[i]))

        observations = [self.observation_fct(i, self.factory) for i in range(len(controllers))]
        rewards = [self.reward_fct(i, self.factory) for i in range(len(controllers))]
        done = all([self.done_fct(i, self.factory) for i in range(len(controllers))])
        info = {}
        return observations, rewards, done, info

    def render(self, mode='human'):
        if mode == 'ansi':
            return factory_string(self.factory)
        elif mode == 'human':
            return print_factory(self.factory)
        else:
            super(FactoryMultiAgentEnv, self).render(mode=mode)

    def reset(self):
        self.factory = deepcopy(self.initial_factory)
