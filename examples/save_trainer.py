"""Start this script with either "deepkit run" or "python examples/save_trainer.py"
"""
from factory.rl import get_config
from factory.environments import register_env_from_config

import ray
import ray.rllib.agents.dqn as algo
from ray.rllib.agents.dqn import DQNTrainer as Trainer
from ray.tune.logger import pretty_print


ray.init(webui_host='127.0.0.1', local_mode=True)
config = get_config(algo)
register_env_from_config()

trainer = Trainer(config=config, env='factory')

for i in range(101):
    result = trainer.train()
    print(pretty_print(result))
    if i % 50 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
