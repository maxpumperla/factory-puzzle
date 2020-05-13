from factory.util.rollout import run
import ray.rllib.agents.dqn as dqn
import os

cls = dqn.DQNTrainer
checkpoint = os.path.expanduser("~/code/factory-puzzle/models/dqn-small-1-1-3/checkpoint_101/checkpoint-101")
steps = 100

run(checkpoint, cls, steps)
