"""Start this script with "python examples/dk_trainer.py"
"""
from factory.environments import *
from factory.rl import get_config

import ray
import ray.rllib.agents.dqn as algo
from ray.rllib.agents.dqn import DQNTrainer as Trainer
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import deepkit


ray.init(webui_host='127.0.0.1')
config = get_config(algo)
print(pretty_print(config))

experiment = deepkit.experiment()
register_env("factory", lambda _: FactoryEnv())
trainer = Trainer(config=config, env="factory")

for i in range(2000):
    result = trainer.train()
    experiment.log_metric("episode_len_mean", result.get("episode_len_mean"))
    experiment.log_metric("episode_reward_max", result.get("episode_reward_max"))
    experiment.log_metric("episode_reward_mean", result.get("episode_reward_mean"))
    experiment.log_metric("episode_reward_min", result.get("episode_reward_min"))

    print(pretty_print(result))
    if i % 50 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
