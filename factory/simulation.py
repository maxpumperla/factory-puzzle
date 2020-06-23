from typing import Optional, List, Dict, Tuple
from collections import Counter
import networkx as nx

import pprint
from .models import *

PRINTER = pprint.PrettyPrinter(indent=2)
VERBOSE = True


def get_shortest_path_with_distance(a,b, factory, obstruction_factor=4):
    distances_and_paths = get_paths_with_distances(a, b, factory, obstruction_factor)
    return distances_and_paths[0]


def get_paths_with_distances(a, b, factory, obstruction_factor=4):
    paths = factory.get_paths(a, b)
    distances = []
    for path in paths:
        # exclude the source from table counting
        tables_on_path = sum([n.has_table() for n in path[1:]])
        # Weighted Manhattan distance: each table on the path contributes "obstruction_factor"
        # all other nodes on the path just 1.
        dist = obstruction_factor * tables_on_path + (len(path) - tables_on_path)
        distances.append(dist)
    paths_and_distances = list(zip(paths, distances))
    return sorted(paths_and_distances, key=lambda pad: pad[1])


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
        self.graph = nx.Graph()

        # Add graph nodes by index
        for node in range(len(self.nodes)):
            self.graph.add_node(node)

        # Graphs are bidirectional by default, so simply add all edges corresponding to neighbours
        for source, node in enumerate(self.nodes):
            for direction in Direction:
                if node.has_neighbour(direction):
                    neighbour = node.get_neighbour(direction)
                    sink = self.nodes.index(neighbour)
                    self.graph.add_edge(source, sink)

        self.paths: Dict[Tuple[int, int], List[int]] = {}

    def get_paths(self, source_node: Node , sink_node: Node):

        # Node to index
        source = self.nodes.index(source_node)
        sink = self.nodes.index(sink_node)

        # If we have a cached result, return it.
        if (source, sink) in self.paths.keys():
            return self.paths.get((source, sink))

        # Otherwise compute all paths and return lists of actual factory nodes, not their indices
        all_paths = list(nx.all_simple_paths(self.graph, source, sink))
        for i, p in enumerate(all_paths):
            all_paths[i] = [self.nodes[n] for n in p]
        # store the result, so we don't have to recompute it later
        self.paths[(source, sink)] = all_paths

        return all_paths

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
