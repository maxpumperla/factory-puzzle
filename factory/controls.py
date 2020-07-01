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


class TableAndRailController:

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

        # Remove table from "start" node
        start.remove_table()

        # Put table on "to" node
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
        if action.value == 4:
            return ActionResult.NONE
        direction = Direction(action.value)
        has_neighbour = table.node.has_neighbour(direction)
        if not has_neighbour:
            return ActionResult.INVALID
        else:
            neighbour = table.node.get_neighbour(direction)
            if neighbour.has_table():
                return ActionResult.COLLISION
            elif neighbour.is_rail:  # node -> rail or rail -> rail
                # can we hop on the rail?
                rail = self.factory.get_rail(node=neighbour)
                assert rail.num_tables() <= 1, "A rail can have at most one table"
                if rail.is_free() or table.node in rail.nodes:
                    return self._move_table(table, neighbour)
                else:
                    # target is blocked with a table.
                    return ActionResult.INVALID
            else:  # Move table from a) node -> node or b) rail -> node
                return self._move_table(table, neighbour)


def move_table_along_path(path: List[Node], factory: Factory):
    if not path[0].has_table():
        raise ValueError("First element of the provided path has to have a table")

    results = []

    for i in range(len(path) - 1):
        node = path[i]
        table = node.table
        if not table:
            break
        # assert table, "No table to move along this path"
        next_node = path[i+1]

        if not node.is_rail and next_node.is_rail:
            rail = factory.get_rail(node=next_node)
            if not rail.is_free():
                # this table obstructs the path of our moving table
                rail_table = rail.get_table_node().table

                all_rail_neighbours = []
                for rail_node in rail.nodes:
                    # all nodes adjacent to this rail that do not lie on the path going forward
                    all_rail_neighbours += list(set((k, n) for k, n in rail_node.neighbours.items()
                                            if n and n not in rail.nodes and n not in path[i:]))

                free_rail_neighbours = [(k, v) for k, v in all_rail_neighbours if not v.has_table()]

                if free_rail_neighbours:
                    # If we have free nodes adjacent to this rail, just pick the first and go there.
                    _, target_node = free_rail_neighbours[0]
                    all_paths = factory.get_unobstructed_paths(rail_table.node, target_node)
                    new_path = all_paths[0]
                    move_table_along_path(new_path, factory)
                else:
                    table_target = None
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

                        table_target = non_rail_node

                    if table_target:
                        rail_paths = factory.get_unobstructed_paths(rail_table.node, table_target)
                        if rail_paths:
                            rail_path = rail_paths[0]
                            move_table_along_path(rail_path, factory)
                        else:
                            raise Exception("Could not move obstacles away for table to enter rail.")
                    else:
                        raise Exception("No candidate found for obstructing table to move to.")

            assert rail.num_tables() <= 1, "At most one table on a rail"

        direction_list = [direction for direction, n in node.neighbours.items() if n == next_node]
        direction = Direction[direction_list[0]]
        action = Action(direction.value)

        result = do_action(table, factory, action)
        factory.add_move(factory.tables.index(table), action, result)

        assert result is not [ActionResult.INVALID, ActionResult.COLLISION]
        results.append(result)

    return results
