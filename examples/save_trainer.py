from factory.environments import *
from factory.util.rl import get_config

import ray
import ray.rllib.agents.dqn as algo
from ray.rllib.agents.dqn import DQNTrainer as Trainer

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

ray.init(webui_host='127.0.0.1')
config = get_config(algo)
print(pretty_print(config))


env_name = "factory"
register_env(env_name, lambda _: FactoryEnv())
trainer = Trainer(config=config, env=env_name)


for i in range(101):
    result = trainer.train()
    print(pretty_print(result))
    if i % 50 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
