from factory.heuristics import solve_factory
from factory.util.samples import get_small_default_factory, get_default_factory


# def test_heuristic_solving():
#     factory = get_small_default_factory(random_seed=1337, num_tables=3, num_cores=3, num_phases=1)
#     solve_factory(factory)
#
#     factory.add_completed_step_count()
#     factory.print_stats()
#
#     assert factory.is_solved()


# def test_heuristic_solving_big_many_cores():
#     factory = get_default_factory(random_seed=7, num_tables=9, num_cores=9, num_phases=3)
#     solve_factory(factory)
#
#     assert factory.is_solved()
#
#     factory.add_completed_step_count()
#     factory.print_stats()


# def test_heuristic_solving_big_many_tables():
#     factory = get_default_factory(random_seed=42, num_tables=12, num_cores=8, num_phases=1)
#
#     solve_factory(factory)
#
#     assert factory.is_solved()
#
#     factory.add_completed_step_count()
#     factory.print_stats()


# def test_heuristic_solving_big_average():
#     failed = 0
#     for i in range(40):
#         factory = get_default_factory(random_seed=i*5, num_tables=8, num_cores=3, num_phases=1)
#         try:
#             solve_factory(factory)
#         except Exception as e:
#             print(i)
#             print(e)
#             failed += 1
#
#     print(failed)
#     assert failed <= 15


