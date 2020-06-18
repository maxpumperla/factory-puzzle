from factory.models import Node, Direction, Table
from factory.simulation import Factory
from factory.config import get_observation_names
import numpy as np
from typing import List
import importlib

__all__ = ["get_observations", "get_reward", "get_done"]


def get_observations(agent_id: int, factory: Factory) -> np.ndarray:
    """Get observation of one agent given the current factory state.
    """
    obs_names = get_observation_names()
    obs_dict = {o: getattr(importlib.import_module('factory.features'), o)(agent_id, factory) for o in obs_names}
    obs_arrays = [obs for obs in obs_dict.values()]
    return np.concatenate(obs_arrays, axis=0)


def get_done(agent_id: int, factory: Factory) -> bool:
    """We're done with the table if it doesn't have a core anymore or we're out of moves.
    """
    if factory.agent_step_counter.get(agent_id) > factory.max_num_steps:
        return True
    agent: Table = factory.tables[agent_id]
    return not agent.has_core()


def get_reward(agent_id: int, factory: Factory) -> float:
    """Get the reward for a single agent in its current state.
    """
    # TODO reward picking as well? (additive terms after all)
    max_num_steps = factory.max_num_steps
    steps = factory.agent_step_counter.get(agent_id)

    agent: Table = factory.tables[agent_id]
    reward = 0.0

    # sum negative rewards due to collisions and illegal moves
    if factory.moves.get(agent_id):
        move = factory.moves[agent_id].pop(-1)
        reward += move.reward()

    # high incentive for reaching a target
    time_taken = max(0, (max_num_steps - steps) / float(max_num_steps))
    if agent.is_at_target:
        reward += 30.0 * max(0, (1 - time_taken))

    # punish if too slow
    # if steps == max_num_steps:
    #     reward -= 100

    # If an agent without core is close to one with core, let it shy away
    # if not agent.has_core():
    #     reward -= has_core_neighbour(agent.node, factory)

    return reward


def one_hot_encode(total: int, positions: List[int]):
    lst = [0 for _ in range(total)]
    for position in positions:
        assert position <= total, "index out of bounds"
        lst[position] = 1
    return lst


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
    return not is_occupied


def has_core_neighbour(node: Node, factory: Factory):
    """If a node has at least one direct neighbour with a core, return True,
    else False. We use this to inform tables without cores to move out of the way
    of tables with cores."""
    for direction in Direction:
        has_direction = node.has_neighbour(direction)
        is_free = check_neighbour(node, direction, factory)
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
        check_neighbour(agent.node, Direction.up, factory),
        check_neighbour(agent.node, Direction.right, factory),
        check_neighbour(agent.node, Direction.down, factory),
        check_neighbour(agent.node, Direction.left, factory)
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