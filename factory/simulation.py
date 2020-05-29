from typing import Optional, List, Dict
from .models import *

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
        self.step_count = 0
        self.moves: Dict[int, List[ActionResult]] = {t: [] for t in range(len(self.tables))}

    def done(self):
        return all([c.done() for c in self.cores])

    def set_tables(self, tables: List[Table]):
        self.tables = tables

    def get_rail(self, node: Node) -> Optional[Rail]:
        for rail in self.rails:
            if node in rail.nodes:
                return rail
        return None

    def add_move(self, agent_id: int, move: ActionResult):
        self.moves.get(agent_id).append(move)
        self.step_count += 1
