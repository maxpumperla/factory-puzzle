import os
import copy
import pickle
from ray.tune.utils import merge_dicts

from factory.environments import FactoryEnv
from ray.rllib.rollout import RolloutSaver, rollout
from ray.tune.registry import register_env

register_env("factory", lambda _: FactoryEnv())
ENV = "factory"
from ray.rllib.models import ModelCatalog
from factory.util.masking import ActionMaskingTFModel, MASKING_MODEL_NAME
ModelCatalog.register_custom_model(MASKING_MODEL_NAME, ActionMaskingTFModel)


def run(checkpoint, cls, steps=1000, out=None, config_args={}):
    config = {}
    # Load configuration from checkpoint file.
    config_dir = os.path.dirname(checkpoint)
    config_path = os.path.abspath(os.path.join(config_dir, "../params.pkl"))
    print(config_path)

    # If no pkl file found, require command line `--config`.
    if not os.path.exists(config_path):
        if not config_args:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory AND no config given on command line!")
    # Load the config from pickled.
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)

    # Set num_workers to be at least 2.
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])

    # Merge with `evaluation_config`.
    evaluation_config = copy.deepcopy(config.get("evaluation_config", {}))
    config = merge_dicts(config, evaluation_config)
    # Merge with command line `--config` settings.
    config = merge_dicts(config, config_args)

    agent = cls(env=ENV, config=config)

    # Load state from checkpoint.
    agent.restore(checkpoint)

    # Do the actual rollout.
    with RolloutSaver(
            outfile=out,
            use_shelve=False,
            write_update_file=False,
            target_steps=steps,
            target_episodes=0,
            save_info=False) as saver:
        rollout(agent, ENV, steps, 0, saver, True, None)
