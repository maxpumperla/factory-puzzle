from .util.samples import get_default_factory, get_small_default_factory


SIMULATION_CONFIG = {
    "layout": "small",
    "num_tables": 3,
    "num_cores": 2,
    "num_phases": 2,
    #"seed": 1337,
    "max_num_steps": 300,
    "actions": 5,
    "observations": 27,
    "low": -1,
    "high": 10,
    "random_init": True,
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
