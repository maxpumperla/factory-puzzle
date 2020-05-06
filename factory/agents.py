"""Agents use the controls of models to find smart ways to act (why).
We'll mainly use this abstraction for testing heuristics and loading RLlib agents."""
from abc import ABC, abstractmethod
from factory.models import Factory, Table, Node
from factory.controls import Action, ActionResult, Controller, TableAndRailController


class Agent(ABC):
    """Agents are the smart parts in the equation. Given a factory
    state, an Agent selects an action, which a Controller can execute
    on their behalf."""

    controller: Controller
    name: str

    def take_action(self, action: Action) -> ActionResult:
        return self.controller.take_action(action)

    @abstractmethod
    def compute_action(self) -> Action:
        """This is where the magic happens. This is modeled after Ray's convention."""
        raise NotImplementedError

    @abstractmethod
    def get_location(self) -> Node:
        raise NotImplementedError

    def save(self):
        pass

    @staticmethod
    def restore():
        pass


class RandomAgent(Agent):
    """Move this table and adjacent shuttles randomly"""

    def __init__(self, table: Table, factory: Factory, name=None):
        controller_name = name + "_controller" if name else None
        self.controller: TableAndRailController = TableAndRailController(table, factory, controller_name)
        self.name = name

    def compute_action(self) -> Action:
        return Action.random_action()

    def get_location(self) -> Node:
        return self.controller.table.node


class Heuristic(Agent):
    """Apply a simple heuristic to get cores to targets."""

    def __init__(self, table: Table, factory: Factory, name=None):
        controller_name = name + "_controller" if name else None
        self.controller: TableAndRailController = TableAndRailController(table, factory, controller_name)
        self.name = name

    def compute_action(self) -> Action:
        return Action.none  # TODO

    def get_location(self) -> Node:
        return self.controller.table.node
