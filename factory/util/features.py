from factory.models import Node, Direction, Table
from factory.simulation import Factory
import numpy as np
from typing import List


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

    return np.asarray(obs)
