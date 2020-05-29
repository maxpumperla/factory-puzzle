from factory.models import Node, Direction, Table
from factory.simulation import Factory
import numpy as np
from typing import List

__all__ = ["get_observations", "get_reward", "get_done"]


def get_observations(agent_id: int, factory: Factory) -> np.ndarray:
    """Get observation of one agent given the current factory state.
    """
    return get_neighbour_observations(agent_id, factory)


def get_done(agent_id: int, factory: Factory) -> bool:
    """We're done with the table if it doesn't have a core anymore or we're out of moves.
    """
    if factory.step_count > factory.max_num_steps:
        return True
    agent: Table = factory.tables[agent_id]
    return not agent.has_core()


def get_reward(agent_id: int, factory: Factory) -> float:
    """Get the reward for a single agent in its current state.
    """
    moves = factory.moves
    max_num_steps = factory.max_num_steps
    steps = factory.step_count

    agent: Table = factory.tables[agent_id]
    reward = 0.0

    # sum negative rewards due to collisions and illegal moves
    reward += sum(m.reward() / 1. for m in moves.get(agent_id))

    # high incentive for reaching a target
    time_taken = max(0, (max_num_steps - steps) / float(max_num_steps))
    if agent.is_at_target:
        reward += 30.0 * (1 - time_taken)

    # If an agent without core is close to one with core, let it shy away
    if not agent.has_core():
        reward -= has_core_neighbour(agent.node, factory)

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


def get_one_hot_observations(agent_id: int, factory: Factory) -> np.ndarray:
    agent: Table = factory.tables[agent_id]

    # Agent node ID one-hot encoded (#Nodes)
    agent_index = [factory.nodes.index(agent.node)]
    num_nodes = len(factory.nodes)
    agent_position = one_hot_encode(num_nodes, agent_index)

    # Current core target one-hot encoded (#Nodes)
    core_target_index = []
    if agent.has_core():
        core_target_index = [factory.nodes.index(agent.core.current_target)]
    target_position = one_hot_encode(num_nodes, core_target_index)

    # Position of all other cores one-hot encoded(#Nodes)
    all_table_indices = [factory.nodes.index(t.node) for t in factory.tables]
    table_positions = one_hot_encode(num_nodes, all_table_indices)

    return np.asarray(agent_position + target_position + table_positions)


def get_neighbour_observations(agent_id: int, factory: Factory) -> np.ndarray:
    # Agent coordinates (2)
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

    # Position of all other cores one-hot encoded(#Nodes = 13 for small, 33 for large)
    num_nodes = len(factory.nodes)
    all_table_indices = [factory.nodes.index(t.node) for t in factory.tables]
    obs += one_hot_encode(num_nodes, all_table_indices)

    return np.asarray(obs)  # 27 small, 47 for large factory
