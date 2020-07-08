from .simulation import Factory, get_paths_distances_obstructions, can_move_away
from .controls import do_action, Action
from .models import Direction, Node, ActionResult
from .util.writer import draw_boxes, load_image
import random
from .util.writer import print_factory
from typing import List
import cv2




def get_core_tables(factory):
    return [t for t in factory.tables if t.has_core()]


def find_suitable_targets(path_list):
    iteration = 0
    search_depth = 1000
    found_solution = False
    while not found_solution and iteration < search_depth:
        iteration += 1
        solution = []
        obstructing_nodes = []
        for i, paths in enumerate(path_list):
            random.shuffle(paths)
            found_path = False
            for path in paths:
                if not found_path:
                    # only add a new path to the solution, if none of the targets of the previous
                    # chosen paths lie on its own trajectory.
                    if all([o not in path for o in obstructing_nodes]):
                        found_path = True
                        solution.append(path)
                        obstructing_nodes.append(path[-1])

        # If we found a set of paths that don't obstruct each other's paths, return them.
        if len(solution) == len(path_list):
            return solution

    return None


def get_path_table_neighbours(path):
    neighbours = []
    for node in path:
        neighbours += [n for n in node.neighbours.values() if n
                       and n not in path and n.has_table()]

    return list(set(neighbours))


def move_away_neighbours(path, factory):
    neighbours = get_path_table_neighbours(path)
    for nb in neighbours:
        good_neighbours = [k for k,n in nb.neighbours.items() if n and n not in path]
        if good_neighbours:
            direction = Direction[good_neighbours[0]]
            action = Action(direction.value)
            do_action(nb.table, factory, action)


def solve_factory(factory: Factory):
    num_phases = sum([t.core.num_phases for t in get_core_tables(factory)])
    max_attempts = num_phases * 50  # explore 50 paths per phase on average
    attempt = 0
    while not factory.is_solved():
        for table in get_core_tables(factory):  # needs to be computed dynamically
            table_phase_solved = False
            start = table.node

            # Note: It might happen that a core is accidentally solved along the way
            if not table.has_core():
                continue

            # If a table is already at its target, complete the phase
            if table.get_target() is table.node:
                table.phase_completed()
                continue

            target = table.core.current_target
            paths = get_paths_distances_obstructions(start, target, factory)

            attempt += 1
            if attempt > max_attempts:
                for t in get_core_tables(factory):
                    print(t.node.coordinates)
                    print(t.core.current_target.coordinates)
                print_factory(factory)
                raise Exception("Number of attempts exceeded.")

            for path, _, _, obstructions, _ in paths:

                # TODO: why does this not work?
                # move_away_neighbours(path, factory)

                table = path[0].table
                if not table_phase_solved:
                    can_all_move = []
                    moving_paths = []
                    for obstructing_table in obstructions:
                        if obstructing_table.node in path:
                            can_move, all_freeing_paths = can_move_away(obstructing_table, path, factory)
                            can_all_move.append(can_move)
                            if can_move:
                                moving_paths.append(all_freeing_paths)

                    solutions = find_suitable_targets(moving_paths)

                    if all(can_all_move) and solutions:

                        # First move away all obstructing tables...
                        for solution in solutions:
                            if solution[0].has_table():
                                move_table_along_path(solution, factory)

                        # ... then move the table to its target, if possible
                        is_path_free_now = len([n for n in path[1:] if n.has_table()]) == 0
                        if is_path_free_now:
                            if path[0] != table.node:
                                # If we accidentally moved the table itself, move it back to its starting point
                                move_table_along_path([table.node, path[0]], factory)
                            if path[0].has_table():
                                move_table_along_path(path, factory)
                                table_phase_solved = True


def move_table_along_path(path: List[Node], factory):
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
