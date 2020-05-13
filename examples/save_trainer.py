from factory.environments import FactoryEnv, RoundRobinFactoryEnv, MultiAgentFactoryEnv
from factory.config import SIMULATION_CONFIG

import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

print(SIMULATION_CONFIG)

ray.init()

config = dqn.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 4
config["eager"] = False

register_env("factory", lambda _: FactoryEnv())
trainer = dqn.DQNTrainer(config=config, env="factory")

for i in range(101):
    result = trainer.train()
    print(pretty_print(result))
    if i % 50 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
