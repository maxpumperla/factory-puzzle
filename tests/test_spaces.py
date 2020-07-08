from factory.environments import get_observation_space
from factory.config import SIMULATION_CONFIG
import numpy as np

def test_obs_space():
    SIMULATION_CONFIG["masking"] = True
    obs_space = get_observation_space(SIMULATION_CONFIG)

    obs = {
        "action_mask": np.zeros(5),
        "observations": np.zeros(obs_space["observations"].shape[0])
    }

    assert obs in obs_space
