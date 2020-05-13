""""Controls specify how objects change state (how)."""
import enum
import random
from abc import ABC, abstractmethod

from .models import Table, Direction, Factory, Node, Rail


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


class ActionResult(enum.IntEnum):
    """Result of an action with attached rewards."""
    NONE = 0,
    MOVED = 0,
    INVALID = -1
    COLLISION = -1

    def reward(self):
        return self.value


class Controller(ABC):

    name: str
    factory: Factory

    @abstractmethod
    def take_action(self, action: Action) -> ActionResult:
        raise NotImplementedError

    def get_factory(self) -> Factory:
        return self.factory


class BaseTableController(Controller):

    def __init__(self, table: Table, factory: Factory, name=None):
        self.table = table
        self.factory = factory
        self.name = name

    def _move_table(self, to: Node) -> None:
        """Move table to an adjacent node. Cores are moved automatically.
        If we move on a rail, also move the shuttle. If the destination
        completes a phase, mark it as such.
        """
        start = self.table.node
        start.remove_table()
        if start.is_rail and to.is_rail:
            start.has_shuttle = False
            to.has_shuttle = True

        self.table.set_node(to)
        to.set_table(self.table)

        if self.table.get_target() is to:
            self.table.phase_completed()
            self.table.is_at_target = True
        else:
            self.table.is_at_target = False

    def _move_to_rail(self, rail, neighbour) -> ActionResult:
        raise NotImplementedError

    def take_action(self, action: Action) -> ActionResult:
        """Attempt to carry out a specified action.
        """
        node = self.table.node
        if action.name == "none":
            # if we don't move, we don't increase the "at target" counter
            self.table.is_at_target = False
            return ActionResult.NONE
        direction = Direction(action.value)
        has_neighbour = node.has_neighbour(direction)
        if not has_neighbour:
            return ActionResult.INVALID
        else:
            neighbour = node.get_neighbour(direction)
            if neighbour.has_table():
                return ActionResult.COLLISION
            elif neighbour.is_rail and not node.is_rail:
                # can we hop on the rail?
                rail = self.factory.get_rail(node=neighbour)
                if rail.is_free():
                    return self._move_to_rail(rail, neighbour)
                else:
                    # target shuttle is blocked with a table.
                    return ActionResult.INVALID
            else:
                # Move table from a) node to node, b) rail to rail or c) rail to node
                self._move_table(neighbour)
                return ActionResult.MOVED


class TableAndRailController(BaseTableController):
    """TableAndRailController controls behaviour of a single `Table` in a `Factory`
    and its adjacent `Rail`s. If the agent wants to move to an available rail, it can
    actively order the respective rail shuttle."""

    def __init__(self, table: Table, factory: Factory, name: str = None):
        super(TableAndRailController, self).__init__(table, factory, name)

    def _move_to_rail(self, rail, neighbour):
        rail.order_shuttle(neighbour)
        self._move_table(neighbour)
        return ActionResult.MOVED


class TableController(BaseTableController):
    """TableController controls behaviour of a single `Table` in a `Factory`.
    The table can only enter a rail, if the shuttle is already right next
    to it (and empty)."""

    def __init__(self, table: Table, factory: Factory, name: str = None):
        super(TableController, self).__init__(table, factory, name)

    def _move_to_rail(self, rail, neighbour):
        if neighbour.has_shuttle:
            self._move_table(neighbour)
            return ActionResult.MOVED
        # shuttle not in position.
        return ActionResult.INVALID


class RailController(Controller):
    """RailController only controls the shuttle on its `Rail`, no tables.
    """

    def __init__(self, rail: Rail, factory: Factory, name: str = None):
        self.rail = rail
        self.factory = factory
        self.name = name

    def take_action(self, action: Action) -> ActionResult:
        node = self.rail.shuttle_node()
        if action.name == "none":
            return ActionResult.NONE
        direction = Direction(action.value)
        has_neighbour = node.has_neighbour(direction)
        if not has_neighbour:
            return ActionResult.INVALID
        else:
            neighbour = node.get_neighbour(direction)
            if neighbour in self.rail.nodes:
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
