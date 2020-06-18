from .util.samples import get_default_factory, get_small_default_factory
import os
import yaml

dk_file_path = "/job/deepkit.yml"
if not os.path.exists(dk_file_path):
    full_dir_name = os.path.dirname(os.path.realpath(__file__))
    dk_file_path = os.path.join(full_dir_name, "../deepkit.yml")

with open(dk_file_path, "r") as f:
    dk_config = yaml.safe_load(f.read()).get("config")

SIMULATION_CONFIG = dk_config
MASK_KEY = "action_mask"
OBS_KEY = "observations"


def get_observation_names():
    return [k for k, v in SIMULATION_CONFIG.items() if k.startswith('obs_') and v is True]


def factory_from_config(config):
    if config is None:
        config = SIMULATION_CONFIG
    if config.get("layout") == "small":
        return get_small_default_factory(**config)
    elif config.get("layout") == "big":
        return get_default_factory(**config)
    else:
        raise ValueError("Choose from either 'small' or 'big'.")

