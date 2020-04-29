import enum
import random
from abc import ABC, abstractmethod

from .models import Table, Direction, Factory, Node


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
    INVALID = -2
    COLLISION = -2

    def reward(self):
        return self.value


class Controller(ABC):

    table: Table = None

    @abstractmethod
    def take_action(self, action: Action) -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def get_factory(self) -> Factory:
        raise NotImplementedError


class TableController(Controller):
    """TableController controls behaviour of a single Table in a Factory. It can move
    in four directions along the specified nodes of the factory. If
    the agent wants to move to an available rail, it can actively order
    the respective rail shuttle."""

    def __init__(self, table: Table, factory: Factory):
        self.table = table
        self.factory = factory

    def get_factory(self) -> Factory:
        return self.factory

    def _move_table(self, to: Node):
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
        assert to.has_shuttle
        to.set_table(self.table)

        if self.table.get_target() is to:
            self.table.phase_completed()
    
    def take_action(self, action: Action) -> ActionResult:
        """Attempt to carry out a specified action.
        """
        node = self.table.node
        if action.name == "none":
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
                    # order shuttle and move table to rail shuttle, if free
                    rail.order_shuttle(neighbour)
                    self._move_table(neighbour)
                    return ActionResult.MOVED
                else:
                    # target shuttle is blocked with a table
                    return ActionResult.INVALID 
            else:
                # Move table from a) node to node, b) rail to rail or c) rail to node
                self._move_table(neighbour)
                return ActionResult.MOVED
