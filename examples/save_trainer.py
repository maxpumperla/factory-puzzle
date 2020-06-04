from factory.environments import *
from factory.util.wrapper import ActionMaskingTFModel, MASKING_MODEL_NAME
from factory.config import SIMULATION_CONFIG

import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog


print(pretty_print(SIMULATION_CONFIG))
ray.init()

config = dqn.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 4
config["eager"] = False

register_env("factory", lambda _: FactoryEnv())

masking = SIMULATION_CONFIG.get("masking")
if masking:
    ModelCatalog.register_custom_model(MASKING_MODEL_NAME, ActionMaskingTFModel)
    config['model'] = {"custom_model": MASKING_MODEL_NAME}
    config['hiddens'] = []
    config['dueling'] = False

trainer = dqn.DQNTrainer(config=config, env="factory")

for i in range(1000):
    result = trainer.train()
    print(pretty_print(result))
    if i % 50 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
