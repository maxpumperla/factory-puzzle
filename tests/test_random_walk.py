from factory.util import get_default_factory, get_small_default_factory, print_factory
from factory.agents import RandomAgent
import cv2
from factory.util.writer import load_image, draw_boxes
from factory.controls import do_action, Action

DEBUG = False


def test_random_completion():
    original_image = load_image(file_name="../assets/large_factory.jpg")

    factory = get_default_factory(random_seed=1337, num_phases=2, num_cores=3, num_tables=8)
    agent = RandomAgent(factory)
    for i in range(40):
        for table in factory.tables:
            do_action(table, factory, Action.random_action())
            factory.step_count += 1
            image = draw_boxes(factory, original_image)
            cv2.imwrite(f"output/factory_{factory.step_count:03}.jpg", image)

    # assert factory.is_solved()


# def test_random_walk():
#     factory = get_small_default_factory(1337)
#     agent = RandomAgent(factory)
#     table = factory.tables[0]
#     for i in range(10):
#         if DEBUG:
#             time.sleep(0.12)
#             print_factory(factory)
#         action = agent.compute_action()
#         if DEBUG:
#             print(">>> Agent location: ", table.node.coordinates)
#             print(">>> Intended action: ", action)
#             print(">>> Result: ", agent.take_action(table, action))
#
#     factory = get_default_factory(123)
#     # time.sleep(2)
#     multi_agent = [RandomAgent(factory) for t in factory.tables]
#     num_agents = len(multi_agent)
#     for i in range(30):
#         if DEBUG:
#             time.sleep(0.05)
#             print_factory(factory)
#         agent = multi_agent[i % num_agents]
#         action = agent.compute_action()
#         if DEBUG:
#             print(">>> Moving multiple agents")
#             print(">>> Result: ", agent.take_action(table, action))
