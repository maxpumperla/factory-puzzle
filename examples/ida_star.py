"""Run me with python examples/ida_star.py
"""
from factory.util.samples import get_small_default_factory
from factory.util import writer
from factory.controls import do_action, Action, ActionResult
from factory.environments import MultiAgentFactoryEnv
import numpy as np
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder, MultiAgentSampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter


batch_builder = SampleBatchBuilder()

writer = JsonWriter("./factory-offline-data")

env = MultiAgentFactoryEnv()

preprocessor = get_preprocessor(env.observation_space)(env.observation_space)


class IDAStar:

    def __init__(self, factory=None):
        # Use a default factory if none provided
        self.factory = get_small_default_factory() if not factory else factory

    def train(self):
        """Train IDA* here"""
        pass

    def smart_step(self) -> Action:
        """"TODO"""
        return {
            0: Action(1),
            1: Action(4),
            # define actions
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
                # for table in self.factory.tables:
                #     result: ActionResult = do_action(table, self.factory, action)

                writer.print_factory(self.factory)

                new_obs, rew, done, info = env.step(action)

                batch_builder.add_values(
                    t=t,
                    eps_id=episode_id,
                    agent_index=0,
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
            writer.write(batch_builder.build_and_reset())



if __name__ == "__main__":
    algo = IDAStar()
    algo.train()
    algo.write_batches()
