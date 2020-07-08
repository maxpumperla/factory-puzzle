from factory.models import Node, Direction, Table
from factory.simulation import Factory
from factory.config import get_observation_names, get_reward_names_and_weights
import numpy as np
from typing import List
import importlib

__all__ = ["get_observations", "get_reward", "get_done", "can_move_in_direction"]


def get_observations(agent_id: int, factory: Factory) -> np.ndarray:
    """Get observation of one agent given the current factory state.
    We first determine the observations selected in the config file and
    then concatenate the results of the corresponding observation functions.

    The names of the functions have to match the names in the config. That's
    also how you "register" a new observation
    """
    obs_names = get_observation_names()
    obs_dict = {obs: getattr(importlib.import_module('factory.features'), obs)(agent_id, factory)
                for obs in obs_names}
    obs_arrays = [obs for obs in obs_dict.values()]
    return np.concatenate(obs_arrays, axis=0)


def get_done(agent_id: int, factory: Factory) -> bool:
    """We're done with the table if it doesn't have a core anymore or we're out of moves.
    """
    counter = factory.agent_step_counter.get(agent_id)
    if counter > factory.max_num_steps:
        # Note that we track the maximum number of steps per agent, not in total.
        return True
    agent: Table = factory.tables[agent_id]
    return not agent.has_core()


def get_reward(agent_id: int, factory: Factory) -> float:
    """Get the reward for a single agent in its current state.
    Similar to observations, reward terms get configured in deepkit.yml.
    """
    rewards = {}
    max_num_steps = factory.max_num_steps
    steps = factory.agent_step_counter.get(agent_id)
    agent: Table = factory.tables[agent_id]

    # sum negative rewards due to collisions and illegal moves
    if factory.moves.get(agent_id):
        move = factory.moves[agent_id].pop(-1)
        rewards["rew_collisions"] = move.reward()

    # high incentive for reaching a target, quickly
    time_taken = steps / float(max_num_steps)
    if agent.is_at_target:
        rewards["rew_found_target"] = (1.0 - time_taken)
        rewards["rew_found_target_squared"] = (1.0 - time_taken) ** 2

    # punish if too slow
    if steps == max_num_steps:
        num_cores_left = len([t for t in factory.tables if t.has_core()])
        rewards["rew_punish_slow_tables"] = - 1 * num_cores_left

    # If an agent without core is close to one with core, let it shy away
    if not agent.has_core():
        rewards["rew_avoid_cores"]  = -1.0 * has_core_neighbour(agent.node, factory)

    # only configured rewards get picked up
    rewards_to_use = get_reward_names_and_weights()

    reward = 0
    for reward_name, weight in rewards_to_use.items():
        # multiply rewards by configured weight terms
        reward += rewards.get(reward_name, 0) * weight

    return reward


def one_hot_encode(total: int, positions: List[int]):
    """Compute one-hot encoding of a list of positions (ones) in
    a vector of length 'total'."""
    lst = [0 for _ in range(total)]
    for position in positions:
        assert position <= total, "index out of bounds"
        lst[position] = 1
    return lst


def can_move_in_direction(node: Node, direction: Direction, factory: Factory):
    """If an agent has a neighbour in the specified direction, add a 1,
    else 0 to the observation space. If that neighbour is free, add 1,
    else 0 (a non-existing neighbour counts as occupied).
    """
    has_direction = node.has_neighbour(direction)
    is_free = False
    if has_direction:
        neighbour: Node = node.get_neighbour(direction)
        if neighbour.is_rail:
            neighbour_rail = factory.get_rail(neighbour)
            is_free = neighbour_rail.is_free() or node in neighbour_rail.nodes
        else:
            is_free = not neighbour.has_table()
    return is_free


def has_core_neighbour(node: Node, factory: Factory):
    """If a node has at least one direct neighbour with a core, return True,
    else False. We use this to inform tables without cores to move out of the way
    of tables with cores."""
    for direction in Direction:
        has_direction = node.has_neighbour(direction)
        is_free = can_move_in_direction(node, direction, factory)
        if has_direction and not is_free:
            neighbour: Node = node.get_neighbour(direction)
            if neighbour.has_table() and neighbour.table.has_core():
                return True
    return False


def obs_agent_id(agent_id: int, factory: Factory) -> np.ndarray:
    return np.asarray([agent_id])


def obs_agent_coordinates(agent_id: int, factory: Factory) -> np.ndarray:
    agent: Table = factory.tables[agent_id]
    return np.asarray(list(agent.node.coordinates))


def obs_all_non_agent_table_coordinates(agent_id: int, factory: Factory) -> np.ndarray:
    """encode all non-agent table coordinates"""
    agent: Table = factory.tables[agent_id]
    coordinates = []
    for table in factory.tables:
        if table is not agent:
            coordinates += list(table.node.coordinates)
    return np.asarray(coordinates)


def obs_agent_has_neighbour(agent_id: int, factory: Factory) -> np.ndarray:
    agent: Table = factory.tables[agent_id]

    return np.asarray([
        agent.node.has_neighbour(Direction.up),
        agent.node.has_neighbour(Direction.right),
        agent.node.has_neighbour(Direction.down),
        agent.node.has_neighbour(Direction.left),
    ])


def obs_agent_free_neighbour(agent_id: int, factory: Factory):
    agent: Table = factory.tables[agent_id]

    return np.asarray([
        can_move_in_direction(agent.node, Direction.up, factory),
        can_move_in_direction(agent.node, Direction.right, factory),
        can_move_in_direction(agent.node, Direction.down, factory),
        can_move_in_direction(agent.node, Direction.left, factory)
    ])


def obs_agent_has_core(agent_id: int, factory: Factory):
    agent: Table = factory.tables[agent_id]
    return np.asarray([agent.has_core()])


def obs_agent_core_target_coordinates(agent_id: int, factory: Factory):
    agent: Table = factory.tables[agent_id]
    if agent.has_core():
        current_target: Node = agent.core.current_target
        return np.asarray(list(current_target.coordinates))
    else:
        return np.asarray([-1, -1])


def obs_all_tables_one_hot(agent_id: int, factory: Factory):
    num_nodes = len(factory.nodes)
    all_table_indices = [factory.nodes.index(t.node) for t in factory.tables]
    return np.asarray(one_hot_encode(num_nodes, all_table_indices))


def obs_all_cores_one_hot(agent_id: int, factory: Factory):
    num_nodes = len(factory.nodes)
    all_table_indices = [factory.nodes.index(t.node) for t in factory.tables if t.has_core()]
    return np.asarray(one_hot_encode(num_nodes, all_table_indices))


def obs_agent_id_one_hot(agent_id: int, factory: Factory):
    agent: Table = factory.tables[agent_id]

    # Agent node ID one-hot encoded (#Nodes)
    agent_index = [factory.nodes.index(agent.node)]
    num_nodes = len(factory.nodes)
    return np.asarray(one_hot_encode(num_nodes, agent_index))


def obs_agent_core_target_one_hot(agent_id: int, factory: Factory) -> np.ndarray:
    agent: Table = factory.tables[agent_id]

    num_nodes = len(factory.nodes)

    # Current core target one-hot encoded (#Nodes)
    core_target_index = []
    if agent.has_core():
        core_target_index = [factory.nodes.index(agent.core.current_target)]
    return np.asarray(one_hot_encode(num_nodes, core_target_index))
