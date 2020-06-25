from .simulation import Factory, get_paths_distances_obstructions, can_move_away
from .controls import move_table_along_path


def get_core_tables(factory):
    return [t for t in factory.tables if t.has_core()]


def solve_factory(factory: Factory):
    tables_with_cores = get_core_tables(factory)
    num_phases = sum([t.core.num_phases for t in tables_with_cores])
    max_attempts = num_phases * 50  # explore 50 paths per phase on average
    attempt = 0
    while not factory.is_solved():
        for table in get_core_tables(factory):
            table_phase_solved = False
            start = table.node
            target = table.core.current_target
            paths = get_paths_distances_obstructions(start, target, factory)

            attempt += 1
            if attempt > max_attempts:
                raise Exception("Number of attempts exceeded.")

            for path, _, _, obstructions, _ in paths:
                if not table_phase_solved:
                    can_all_move = []
                    moving_paths = []
                    for obstructing_table in obstructions:
                        can_move, moving_path = can_move_away(obstructing_table, path, factory)
                        can_all_move.append(can_move)
                        moving_paths.append(moving_path)
                    if all(can_all_move):
                        # TODO: problem if tables block each other on their way out
                        #  First check if all CAN move away, then actually find ways to do so?
                        #  Could choose paths so that the current path's target is not on the route of the next

                        # Move away all obstructing tables...
                        for mp in moving_paths:
                            move_table_along_path(mp, factory)
                        # ... then move the table to its target
                        move_table_along_path(path, factory)
                        table_phase_solved = True

