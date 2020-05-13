"""In which environment are things happening? (where)
The environment specifies what agents can observe and how
they are rewarded for actions."""
from factory.models import Factory, Table, Direction, Node
from factory.controls import TableController, Action, ActionResult
from factory.util import print_factory, factory_string
from factory.config import SIMULATION_CONFIG, get_factory_from_config

from copy import deepcopy

import gym
from gym import spaces
from ray import rllib
import numpy as np
from typing import Callable, List, Dict


class StatisticsTracker:

    def __init__(self, factory: Factory, max_num_steps):
        self.max_num_steps = max_num_steps
        self.factory = factory
        self.step_count = 0
        self.moves: Dict[int, List[ActionResult]] = {t: [] for t in range(len(factory.tables))}

    def add_move(self, agent_id: int, move: ActionResult):
        self.moves.get(agent_id).append(move)
        self.step_count += 1

    @staticmethod
    def from_config(config):
        factory = get_factory_from_config(config)
        max_num_steps = config.get("max_num_steps")
        return StatisticsTracker(factory, max_num_steps)


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


def has_core_neighbour(node: Node, factory: Factory):
    """If a node has at least one direct neighbour with a core, return True,
    else False. We use this to inform tables without cores to move out of the way
    of tables with cores."""
    for direction in Direction:
        has_dir, is_free = check_neighbour(node, direction, factory)
        if has_dir and not is_free:
            neighbour: Node = node.get_neighbour(direction)
            if neighbour.has_table() and neighbour.table.has_core():
                return True
    return False


def get_observations(agent_id: int, factory: Factory) -> np.ndarray:
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


def get_reward(agent_id: int, factory: Factory, tracker: StatisticsTracker) -> float:
    """Get the reward for a single agent in its current state.
    """
    moves = tracker.moves
    max_num_steps = tracker.max_num_steps
    steps = tracker.step_count

    agent: Table = factory.tables[agent_id]
    reward = 0.0

    # sum negative rewards due to collisions and illegal moves
    reward += sum(m.reward() / 200. for m in moves.get(agent_id))

    # high incentive for reaching a target
    time_taken = (max_num_steps - steps) / float(max_num_steps)
    if agent.is_at_target:
        reward += 10.0 * (1 - time_taken)

    # If an agent without core is close to one with core, let it shy away
    if not agent.has_core():
        reward -= has_core_neighbour(agent.node, factory)

    return reward


def get_done(agent_id: int, factory: Factory, tracker: StatisticsTracker) -> bool:
    """We're done with the table if it doesn't have a core anymore or we're out of moves.
    """
    agent: Table = factory.tables[agent_id]
    return not agent.has_core() or tracker.step_count > tracker.max_num_steps


class FactoryEnv(gym.Env):
    """Define a simple OpenAI Gym environment for a single agent."""

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, config=None):
        if config is None:
            config = SIMULATION_CONFIG
        self.config = config
        self.tracker = StatisticsTracker.from_config(self.config)
        self.factory = self.tracker.factory
        self.initial_factory = deepcopy(self.factory)

        self.num_actions = config.get("actions")
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Box(
            low=config.get("low"),
            high=config.get("high"),
            shape=(config.get("observations"),),
            dtype=np.float32
        )

    def step(self, action):
        assert action in range(self.num_actions)

        agent_id = 0
        table = self.factory.tables[agent_id]
        controller = TableController(table, self.factory)
        action_result = controller.take_action(Action(action))
        self.tracker.add_move(agent_id, action_result)

        observations: np.ndarray = get_observations(agent_id, self.factory)
        rewards = get_reward(agent_id, self.factory, self.tracker)
        done = get_done(agent_id, self.factory, self.tracker)
        return observations, rewards, done, {}

    def render(self, mode='human'):
        if mode == 'ansi':
            return factory_string(self.factory)
        elif mode == 'human':
            return print_factory(self.factory)
        else:
            super(FactoryEnv, self).render(mode=mode)

    def reset(self):
        # TODO later on make this reset to a random factory with same layout
        #  by using: self.factory = get_factory_from_config(self.config)
        self.factory = deepcopy(self.initial_factory)
        return get_observations(0, self.factory)


class RoundRobinFactoryEnv(FactoryEnv):

    def __init__(self, config=None):
        super(RoundRobinFactoryEnv, self).__init__(config)
        self.current_agent = 0
        self.num_agents = self.config.get("num_tables")

    def step(self, action):
        assert action in range(self.num_actions)

        table = self.factory.tables[self.current_agent]
        controller = TableController(table, self.factory)
        action_result = controller.take_action(Action(action))
        self.tracker.add_move(self.current_agent, action_result)

        observations: np.ndarray = get_observations(self.current_agent, self.factory)
        rewards = get_reward(self.current_agent, self.factory, self.tracker)
        done = get_done(self.current_agent, self.factory, self.tracker)

        self.current_agent = (self.current_agent + 1) % self.num_agents

        return observations, rewards, done, {}


class MultiAgentFactoryEnv(rllib.env.MultiAgentEnv):
    """Define a ray multi agent env"""

    def __init__(self, config=None):
        if config is None:
            config = SIMULATION_CONFIG
        self.config = config
        self.tracker = StatisticsTracker.from_config(self.config)
        self.factory = self.tracker.factory
        self.initial_factory = deepcopy(self.factory)
        self.num_agents = self.config.get("num_tables")

        self.num_actions = config.get("actions")
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Box(
            low=config.get("low"),
            high=config.get("high"),
            shape=(config.get("observations"),),
            dtype=np.float32
        )

    def step(self, action: Dict):
        tables = self.factory.tables
        controllers = [TableController(t, self.factory) for t in tables]

        keys = action.keys()
        for i in keys:
            assert action.get(i) is not None, action
            controllers[i].take_action(Action(action.get(i)))

        observations = {i: get_observations(i, self.factory) for i in keys}
        rewards = {i: get_reward(i, self.factory, self.tracker) for i in keys}
        dones = {i: get_done(i, self.factory, self.tracker) for i in keys}
        all_done = all(v for k, v in dones.items())
        dones['__all__'] = all_done

        return observations, rewards, dones, {}

    def render(self, mode='human'):
        if mode == 'ansi':
            return factory_string(self.factory)
        elif mode == 'human':
            return print_factory(self.factory)
        else:
            super(MultiAgentFactoryEnv, self).render(mode=mode)

    def reset(self):
        self.factory = deepcopy(self.initial_factory)
        return {i: get_observations(i, self.factory) for i in range(self.num_agents)}
