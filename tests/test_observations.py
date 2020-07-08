from factory.features import get_observations
from factory.util.samples import get_small_default_factory


def test_get_obs():
    factory = get_small_default_factory()
    agent_id = 0
    obs = get_observations(agent_id, factory)
