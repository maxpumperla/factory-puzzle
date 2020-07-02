"""Agents use the controls of models to find smart ways to act (why).
We'll mainly use this abstraction for testing heuristics and loading RLlib agents."""
from abc import ABC, abstractmethod
import os
import pickle
import numpy as np
from factory.models import Table
from factory.simulation import Factory
from factory.controls import Action, ActionResult, TableAndRailController


class Agent(ABC):
    """Agents are the smart parts in the equation. Given a factory
    state, an Agent selects an action, which a Controller can execute
    on their behalf."""

    controller: TableAndRailController
    name: str

    def take_action(self, table: Table, action: Action) -> ActionResult:
        return self.controller.do_action(table, action)

    @abstractmethod
    def compute_action(self, observations: np.ndarray) -> Action:
        """This is where the magic happens. This is modeled after Ray's convention."""
        raise NotImplementedError

    def save(self):
        pass

    @staticmethod
    def restore():
        pass


class RandomAgent(Agent):
    """Move this table and adjacent shuttles randomly"""

    def __init__(self, factory: Factory, name=None):
        controller_name = name + "_controller" if name else None
        self.controller: TableAndRailController = TableAndRailController(factory, controller_name)
        self.name = name

    def compute_action(self, observations=None) -> Action:
        return Action.random_action()


class RayAgent(Agent):
    """Move this table and adjacent shuttles according to a Ray RLlib policy"""
    def __init__(self, factory: Factory, env_name: str, policy_file_name: str,
                 agent_cls, name=None):
        controller_name = name + "_controller" if name else None
        self.controller: TableAndRailController = TableAndRailController(factory, controller_name)
        self.name = name

        config_dir = os.path.dirname(policy_file_name)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        agent = agent_cls(env=env_name, config=config)
        agent.restore(policy_file_name)

        self.agent = agent

    def compute_action(self, observations) -> Action:
        return Action(self.agent.compute_action(observations))


class HeuristicAgent(Agent):
    """Apply a simple heuristic to get cores to targets."""

    def __init__(self, factory: Factory, name=None):
        controller_name = name + "_controller" if name else None
        self.controller: TableAndRailController = TableAndRailController(factory, controller_name)
        self.name = name

    def compute_action(self, observations) -> Action:
        return Action.none  # TODO
