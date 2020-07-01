""""Controls specify how objects change state (how)."""
import enum
import random
from typing import List

from .models import Table, Direction, Node, Rail, ActionResult
from .simulation import Factory


class Action(enum.IntEnum):
    """Move in a direction or stay where you are."""
    up = 0
    right = 1
    down = 2
    left = 3
    none = 4

    @staticmethod
    def random_action():
        return Action(random.randrange(0, 5))


def do_action(table: Table, factory: Factory, action: Action):
    return TableAndRailController(factory).do_action(table, action)


def move_table_along_path(path: List[Node], factory: Factory):
    if not  path[0].has_table():
        raise ValueError("First element of the provided path has to have a table")

    results = []

    for i in range(len(path) - 1):
        node = path[i]
        table = node.table
        assert table, "No table to move along this path"
        next_node = path[i+1]

        if not node.is_rail and next_node.is_rail:
            rail = factory.get_rail(node=next_node)
            if not rail.is_free():
                # this shuttle obstructs the path of our moving table
                shuttle = rail.shuttle_node()

                all_rail_neighbours = []
                for rail_node in rail.nodes:
                    # all nodes adjacent to this rail that do not lie on the path going forward
                    all_rail_neighbours += list(set((k, n) for k, n in rail_node.neighbours.items()
                                            if n and n not in rail.nodes and n not in path[i:]))

                free_rail_neighbours = [(k, v) for k, v in all_rail_neighbours if not v.has_table()]

                # The rail shuttle is occupied and needs to be emptied
                shuttle_target = None

                if free_rail_neighbours:
                    # If we have free nodes adjacent to this rail, just pick the first and go there.
                    _, target_node = free_rail_neighbours[0]
                    all_paths = factory.get_unobstructed_paths(shuttle, target_node)
                    new_path = all_paths[0]
                    move_table_along_path(new_path, factory)
                else:
                    for _, non_rail_node in all_rail_neighbours:
                        free_adjacent_neighbours = [k for k, v in non_rail_node.neighbours.items()
                                                    if v and not v.has_table()]
                        if free_adjacent_neighbours:
                            direction = Direction[free_adjacent_neighbours[0]]
                            action = Action(direction.value)
                            this_table = non_rail_node.table
                            result = do_action(this_table, factory, action)
                            assert result is not [ActionResult.INVALID, ActionResult.COLLISION]
                            factory.add_move(factory.tables.index(this_table), action, result)

                        shuttle_target = non_rail_node

                    rail_paths = factory.get_unobstructed_paths(shuttle, shuttle_target)
                    if rail_paths:
                        rail_path = rail_paths[0]
                        move_table_along_path(rail_path, factory)
                    else:
                        raise Exception("Could not move obstacles away for table to enter rail.")

            assert len([n for n in rail.nodes if n.has_table()]) <= 1, "At most one table on a rail"
            assert len([n for n in rail.nodes if n.has_shuttle]) == 1, "Exactly one rail node is the shuttle"


        direction_list = [direction for direction, n in node.neighbours.items() if n == next_node]
        direction = Direction[direction_list[0]]
        action = Action(direction.value)

        result = do_action(table, factory, action)
        factory.add_move(factory.tables.index(table), action, result)

        assert result is not [ActionResult.INVALID, ActionResult.COLLISION]
        results.append(result)

    return results

class BaseTableController:

    def __init__(self, factory: Factory, name=None):
        self.factory = factory
        self.name = name

    @staticmethod
    def _move_table(table: Table, to: Node) -> ActionResult:
        """Move table to an adjacent node. Cores are moved automatically.
        If we move on a rail, also move the shuttle. If the destination
        completes a phase, mark it as such.
        """
        start = table.node
        start.remove_table()
        if start.is_rail and to.is_rail:
            # assert start.has_shuttle, "To move along a rail, the table has to be on a shuttle"
            start.has_shuttle = False
            to.has_shuttle = True

        table.set_node(to)
        to.set_table(table)

        if table.get_target() is to:
            table.phase_completed()
            table.is_at_target = True
        else:
            table.is_at_target = False
        return ActionResult.MOVED

    def _move_to_rail(self, table: Table, rail: Rail, neighbour: Node) -> ActionResult:
        raise NotImplementedError

    def do_action(self, table: Table, action: Action) -> ActionResult:
        """Attempt to carry out a specified action.
        """
        table.is_at_target = False  # Reset target
        node = table.node
        if action.value == 4:
            return ActionResult.NONE
        direction = Direction(action.value)
        has_neighbour = node.has_neighbour(direction)
        if not has_neighbour:
            return ActionResult.INVALID
        else:
            neighbour = node.get_neighbour(direction)
            if neighbour.has_table():
                return ActionResult.COLLISION
            elif neighbour.is_rail and not node.is_rail:  # node -> rail
                # can we hop on the rail?
                rail = self.factory.get_rail(node=neighbour)
                if rail.is_free():
                    return self._move_to_rail(table, rail, neighbour)
                else:
                    # target shuttle is blocked with a table.
                    return ActionResult.INVALID
            else:  # Move table from a) node -> node, b) rail -> rail or c) rail -> node
                return self._move_table(table, neighbour)


class TableAndRailController(BaseTableController):
    """TableAndRailController controls behaviour of a single `Table` in a `Factory`
    and its adjacent `Rail`s. If the agent wants to move to an available rail, it can
    actively order the respective rail shuttle."""

    def __init__(self, factory: Factory, name: str = None):
        super(TableAndRailController, self).__init__(factory, name)

    def _move_to_rail(self, table: Table, rail: Rail, neighbour: Node):
        rail.order_shuttle(neighbour)
        return self._move_table(table, neighbour)


class TableController(BaseTableController):
    """TableController controls behaviour of a single `Table` in a `Factory`.
    The table can only enter a rail, if the shuttle is already right next
    to it (and empty)."""

    def __init__(self, factory: Factory, name: str = None):
        super(TableController, self).__init__(factory, name)

    def _move_to_rail(self, table: Table, rail: Rail, neighbour: Node):
        if neighbour.has_shuttle:
            return self._move_table(table, neighbour)
        # shuttle not in position.
        return ActionResult.INVALID


class RailController:
    """RailController only controls the shuttle on its `Rail`, no tables.
    """

    def __init__(self, factory: Factory, name: str = None):
        self.factory = factory
        self.name = name

    @staticmethod
    def take_action(rail: Rail, action: Action) -> ActionResult:
        node = rail.shuttle_node()
        if action.value == 4:
            return ActionResult.NONE
        direction = Direction(action.value)
        has_neighbour = node.has_neighbour(direction)
        if not has_neighbour:
            return ActionResult.INVALID
        else:
            neighbour = node.get_neighbour(direction)
            if neighbour in rail.nodes:
                node.has_shuttle = False
                neighbour.has_shuttle = True
                if node.has_table():
                    table = node.table
                    table.set_node(neighbour)
                    assert not node.has_table()
                    neighbour.set_table(table)
                return ActionResult.MOVED
            else:
                return ActionResult.INVALID
