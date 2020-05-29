from factory.util.samples import get_small_default_factory
from factory.controls import ActionResult
from factory.agents import RandomAgent


def test_stats_tracker():
    factory = get_small_default_factory(random_seed=42, num_tables=4, num_cores=1, num_phases=1, max_num_steps=1000)

    agent_id = 0
    table = factory.tables[agent_id]
    agent = RandomAgent(factory)

    for i in range(100):
        action = agent.compute_action()
        action_result = agent.take_action(table, action)
        factory.add_move(agent_id, action_result)
        assert factory.step_count < factory.max_num_steps

    assert sum(len(v) for k, v in factory.moves.items()) == 100
    for result in ActionResult:
        # expect every kind of movement
        assert result in factory.moves.get(agent_id)

    rewards = sum(res.reward() for res in factory.moves.get(agent_id))
    invalid = len([m for m in factory.moves.get(agent_id) if m is ActionResult.INVALID])
    collisions = len([m for m in factory.moves.get(agent_id) if m is ActionResult.COLLISION])

    # negative rewards are the sum of invalids and collisions.
    assert invalid + collisions + rewards == 0
