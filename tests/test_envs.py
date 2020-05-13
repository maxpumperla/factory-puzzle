from factory.util.samples import get_small_default_factory
from factory.environments import StatisticsTracker
from factory.controls import ActionResult
from factory.agents import RandomAgent


def test_stats_tracker():
    factory = get_small_default_factory(random_seed=42, num_tables=4, num_cores=1, num_phases=1)
    tracker = StatisticsTracker(factory, 1000)

    agent_id = 0
    table = factory.tables[agent_id]
    agent = RandomAgent(table, factory)

    for i in range(100):
        action = agent.compute_action()
        action_result = agent.take_action(action)
        tracker.add_move(agent_id, action_result)
        assert tracker.step_count < tracker.max_num_steps

    assert sum(len(v) for k, v in tracker.moves.items()) == 100
    for result in ActionResult:
        assert result in tracker.moves.get(agent_id)

    rewards = sum(res.reward() for res in tracker.moves.get(agent_id))
    invalid = len([m for m in tracker.moves.get(agent_id) if m is ActionResult.INVALID])
    collisions = len([m for m in tracker.moves.get(agent_id) if m is ActionResult.COLLISION])

    assert invalid + collisions + rewards == 0
