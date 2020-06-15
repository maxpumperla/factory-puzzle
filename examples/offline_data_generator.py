from factory.environments import FactoryEnv
import numpy as np

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder, MultiAgentSampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter


batch_builder = SampleBatchBuilder()
writer = JsonWriter("./factory-offline-data")

env = FactoryEnv()

prep = get_preprocessor(env.observation_space)(env.observation_space)

for eps_id in range(100):
    observations = env.reset()
    prev_action = np.zeros_like(env.action_space.sample())
    num_actions = 5
    action_dist_inputs = np.ones(num_actions)

    prev_reward = 0
    done = False
    t = 0
    while not done:
        # TODO use a heuristic here instead
        action = env.action_space.sample()

        new_obs, rew, done, info = env.step(action)
        batch_builder.add_values(
            t=t,
            eps_id=eps_id,
            agent_index=0,
            obs=prep.transform(observations),
            actions=action,
            action_prob=1.0,  # put the true action probability here
            # action_logp=0.0,  # log probabilities
            # action_dist_inputs=action_dist_inputs,
            rewards=rew,
            prev_actions=prev_action,
            prev_rewards=prev_reward,
            dones=done,
            infos=info,
            new_obs=prep.transform(new_obs))
        observations = new_obs
        prev_action = action
        prev_reward = rew
        t += 1
    writer.write(batch_builder.build_and_reset())
