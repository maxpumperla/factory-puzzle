"""Run me with python examples/ida_star.py
"""
from factory.util.samples import get_small_default_factory
from factory.util import writer
from factory.controls import do_action, Action, ActionResult
from factory.environments import MultiAgentFactoryEnv
from factory.models import Table, Node
from factory.simulation import get_shortest_path_with_distance

import numpy as np
from typing import List, Dict
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import MultiAgentSampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter


multi_batch_builder = MultiAgentSampleBatchBuilder()
writer = JsonWriter("./factory-offline-data")

env = MultiAgentFactoryEnv()

preprocessor = get_preprocessor(env.observation_space)(env.observation_space)


class IDAStar:

    def __init__(self, factory=None):
        # Use a default factory if none provided
        self.factory = get_small_default_factory() if not factory else factory
        self.obstruction_factor = 2
        self.max_bound = 30

    def get_cost_bound(self, source: Node, target: Node):
        # Get a lower bound on cost for a table to reach its target
        shortest_path, distance = get_shortest_path_with_distance(source, target, self.factory, self.obstruction_factor)
        return distance

    def find_path(self, root: Table):
        """Main method of IDA*, which finds a feasible path of a table to its target or returns None"""
        if not root.has_core():
            raise Exception("Can only compute solutions for tables with a core.")

        target: Node = root.core.current_target
        h = self.get_cost_bound(root.node, target)
        g = 0
        path = [root.node]

        while True:
            found_solution, path, h = self.search(path, target, g, h)
            if found_solution:
                return path, h
            if h >= self.max_bound:
                return None

    def search(self, path: List, target: Node, g: int, h: int):
        """Iterative search method for IDA*"""
        last_node = path[-1]
        f = g + self.get_cost_bound(last_node, target)

        # set new bound if current estimate is higher
        if f > h:
            return False, path, f

        # If we found the target, return solution and estimate
        if last_node == target:
            return True, path, h

        # Set minimum to a predefined upper bound (instead of np.inf, in which case it might not converge)
        minimum = self.max_bound

        # Iterate through all neighbours that are not already in the current path
        successors = [v for k, v in last_node.neighbours if v and v not in path]
        for successor in successors:
            # TODO: appending works fine, but what do we do about the factory state?
            #  it's currently unclear how to "move away" an obstructing table, as we cannot simply pass it.
            #  this needs to be fixed somehow
            path.append(successor)
            # Search iteratively
            found, path, new_h = self.search(path, target, g + 1, h)
            if found:
                return found, path, new_h
            minimum = min(minimum, new_h)
            path.pop()

        return False, path, minimum

    def smart_step(self) -> Dict[int, Action]:
        """"TODO"""
        return {
            0: Action(1),
            1: Action(4),
            # define actions according to IDA*
        }


    def write_batches(self, num_episodes=100):
        for episode_id in range(num_episodes):

            observations = env.reset()
            prev_action = np.zeros_like(env.action_space.sample())

            prev_reward = 0
            done = False
            t = 0
            while not done:
                action = self.smart_step()

                writer.print_factory(self.factory)

                new_obs, rew, done, info = env.step(action)

                # TODO: example from GitHub was for gym.Env, we need a MultiAgentEnv batch builder.
                multi_batch_builder.add_values(
                    t=t,
                    eps_id=episode_id,
                    agent_index=0, # TODO: this is multiagent, what do we do here?
                    obs=preprocessor.transform(observations),
                    actions=action,
                    action_prob=1.0,  # put the true action probability here
                    rewards=rew,
                    prev_actions=prev_action,
                    prev_rewards=prev_reward,
                    dones=done,
                    infos=info,
                    new_obs=preprocessor.transform(new_obs))
                observations = new_obs
                prev_action = action
                prev_reward = rew
                t += 1
            writer.write(multi_batch_builder.build_and_reset())



if __name__ == "__main__":
    algo = IDAStar()
    # TODO: cycle through all tables with cores and solve for them, then write batches.
    algo.write_batches()
