import ray
from factory.util.rollout import run
import ray.rllib.agents.dqn as dqn
import os

# TODO only works with this model if len(obs) = 14
# def test_rollouts():
#     ray.init()
#     cls = dqn.DQNTrainer
#     checkpoint = os.path.abspath("tests/resources/dqn-small-1-1-3/checkpoint_101/checkpoint-101")
#     steps = 100
#
#     run(checkpoint, cls, steps)
#     ray.shutdown()
