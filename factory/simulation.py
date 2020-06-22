from typing import Optional, List, Dict
from collections import Counter
import pprint
from .models import *

PRINTER = pprint.PrettyPrinter(indent=2)
VERBOSE = True


class Factory:
    """A Factory sets up all components (nodes, rails, tables) needed to
    solve the problem of delivering cores to their destinations. Note that
    this is just a "model", the application logic and agent interaction is
    separated  """
    def __init__(self, nodes: List[Node], rails: List[Rail],
                 tables: List[Table], max_num_steps: int  = 1000, name: str = None):
        self.nodes = nodes
        self.rails = rails
        self.tables = tables
        self.name = name
        self.cores = [t.core for t in self.tables if t.has_core()]
        self.max_num_steps = max_num_steps

        # Stats counter
        self.step_count = 0
        self.agent_step_counter: Dict[int, int] = {t: 0 for t in range(len(self.tables))}
        self.moves: Dict[int, List[ActionResult]] = {t: [] for t in range(len(self.tables))}
        self.move_counter = Counter()
        self.action_counter = Counter()
        self.step_completion_counter: Dict[int, List[int]] = {t: [] for t in range(len(self.tables))}

    def done(self):
        return all([c.done() for c in self.cores])

    def set_tables(self, tables: List[Table]):
        self.tables = tables

    def get_rail(self, node: Node) -> Optional[Rail]:
        for rail in self.rails:
            if node in rail.nodes:
                return rail
        return None

    def add_move(self, agent_id: int, action, move: ActionResult):
        self.step_count += 1
        self.moves.get(agent_id).append(move)
        self.agent_step_counter[agent_id] += 1
        self.move_counter[move.name]  += 1
        self.action_counter[action.name] +=1

    def add_completed_step_count(self):
        for agent_id in range(len(self.tables)):
            counter = self.step_completion_counter.get(agent_id)
            counter.append(self.agent_step_counter[agent_id])

    def print_stats(self):
        if VERBOSE:
            PRINTER.pprint(">>> Completed an episode")
            # for core in self.cores:
            #     PRINTER.pprint(core.table.node.coordinates)
            # for table in self.tables:
            #     PRINTER.pprint(table.has_core())
            PRINTER.pprint("   >>> Move counter")
            PRINTER.pprint(dict(self.move_counter))
            PRINTER.pprint("   >>> Action counter")
            PRINTER.pprint(dict(self.action_counter))
            PRINTER.pprint("   >>> Steps taken to completion")
            PRINTER.pprint(self.step_completion_counter)
