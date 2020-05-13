from .models import Factory
from .util.samples import get_default_factory, get_small_default_factory
from typing import Dict

SIMULATION_CONFIG = {
    "layout": "small",
    "num_tables": 1,
    "num_cores": 1,
    "num_phases": 1,
    "seed": 1337,
    "max_num_steps": 1000,
    "actions": 5,
    "observations": 14,
    "low": -1,
    "high": 7
}


def get_factory_from_config(config: Dict = None) -> Factory:
    if config is None:
        config = SIMULATION_CONFIG
    if config.get("layout") == "small":
        return get_small_default_factory(**config)
    elif config.get("layout") == "big":
        return get_default_factory(**config)
    else:
        raise ValueError("Choose from either 'small' or 'big'.")
