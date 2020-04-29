from abc import ABC, abstractmethod
from factory.models import Factory, Table, Node
from factory.controls import Action, ActionResult, Controller, TableController


class Agent(ABC):
    """Agents are the smart parts in the equation. Given a factory
    state, an Agent selects an action, which a Controller can execute
    on their behalf"""

    controller: Controller = None

    @abstractmethod
    def compute_action(self) -> Action:
        """This is where the magic happens. This is modeled after Ray's convention."""
        raise NotImplementedError

    @abstractmethod
    def get_location(self) -> Node:
        raise NotImplementedError

    def take_action(self, action: Action) -> ActionResult:
        return self.controller.take_action(action)


class RandomTableAgent(Agent):
    """Move this table randomly"""

    def __init__(self, table: Table, factory: Factory):
        self.controller = TableController(table, factory)

    def compute_action(self) -> Action:
        return Action.random_action()

    def get_location(self) -> Node:
        return self.controller.table.node
