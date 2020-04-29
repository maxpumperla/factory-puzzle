from factory.util import get_default_factory, print_factory
from factory.agents import RandomAgent
from factory.environments import FactoryEnv
import time

if __name__ == "__main__":
    factory = get_default_factory(123)
    agent = RandomAgent(factory.tables[0], factory)
    env = FactoryEnv(factory, 5)
    for i in range(100):
        time.sleep(0.2)
        print_factory(factory)
        action = agent.compute_action()
        print(">>> Agent location: ", agent.get_location().coordinates)
        print(">>> Intended action: ", action)
        print(">>> Result: ", agent.take_action(action))

    time.sleep(2)
    multi_agent = [RandomAgent(t, factory) for t in factory.tables]
    num_agents = len(multi_agent)
    for i in range(300):
        time.sleep(0.05)
        print_factory(factory)
        agent = multi_agent[i % num_agents]
        action = agent.compute_action()
        print(">>> Moving multiple agents")
        print(">>> Result: ", agent.take_action(action))
