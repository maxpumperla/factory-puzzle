from factory.util.samples import get_small_default_factory
from factory.controls import ActionResult, Action
from factory.agents import RandomAgent
from factory.config import SIMULATION_CONFIG
from factory.environments import FactoryEnv
from copy import deepcopy


def test_stats_tracker():
    factory = get_small_default_factory(random_seed=42, num_tables=4, num_cores=1, num_phases=1, max_num_steps=1000)

    agent_id = 0
    table = factory.tables[agent_id]
    agent = RandomAgent(factory)

    for i in range(100):
        action = agent.compute_action()
        action_result = agent.take_action(table, action)
        factory.add_move(agent_id, Action(action), action_result)
        assert factory.agent_step_counter.get(agent_id) < factory.max_num_steps

    assert sum(len(v) for k, v in factory.moves.items()) == 100
    for result in ActionResult:
        # expect every kind of movement, except invalidly entering a rail
        if result is not ActionResult.INVALID_RAIL_ENTERING:
            assert result in factory.moves.get(agent_id)

    rewards = sum(res.reward() for res in factory.moves.get(agent_id))
    invalid = len([m for m in factory.moves.get(agent_id) if m is ActionResult.INVALID])
    collisions = len([m for m in factory.moves.get(agent_id) if m is ActionResult.COLLISION])

    # negative rewards are the sum of invalids and collisions.
    assert invalid + collisions + rewards == 0


def test_env_reset():
    config = SIMULATION_CONFIG
    config['random_init'] = False
    config['env'] = "FactoryEnv"
    env = FactoryEnv(config=config)
    factory = env.factory
    initial_factory = deepcopy(factory)

    # Alter the env
    factory.tables.pop()

    env.reset()
    for a,b in zip(env.factory.tables, initial_factory.tables):
        assert a.node.coordinates == b.node.coordinates


