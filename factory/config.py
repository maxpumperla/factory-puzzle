from .util.samples import get_default_factory, get_small_default_factory


SIMULATION_CONFIG = {
    "layout": "small",
    "num_tables": 1,
    "num_cores": 1,
    "num_phases": 1,
    #"seed": 1337,
    "max_num_steps": 300,
    "actions": 5,
    "observations": 14, #27
    "low": -1,
    "high": 10,
    "random_init": False,
    "masking": True,
}


def factory_from_config(config):
    if config is None:
        config = SIMULATION_CONFIG
    if config.get("layout") == "small":
        return get_small_default_factory(**config)
    elif config.get("layout") == "big":
        return get_default_factory(**config)
    else:
        raise ValueError("Choose from either 'small' or 'big'.")

