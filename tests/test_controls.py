from factory.models import Factory, Rail, Node, Table, Core, Direction, Phase
from factory.controls import TableAndRailController, Action, ActionResult


def test_table_controller():
    """S -> a -> b -> c -> T
       |         s
       C
    """
    start, target, collision = Node("S"), Node("T"), Node("C")
    node_a, node_b, node_c = Node("a", is_rail=True), Node("b", is_rail=True), Node("c", is_rail=True)
    start.add_neighbour(node_a, Direction.right)
    start.add_neighbour(collision, Direction.down)
    node_a.add_neighbour(node_b, Direction.right)
    node_b.add_neighbour(node_c, Direction.right)
    node_c.add_neighbour(target, Direction.right)
    nodes = [start, node_a, node_b, node_c, target]

    rail = Rail(nodes=[node_a, node_b, node_c], shuttle=node_b)
    rails = [rail]

    table = Table(start)
    phases = {Phase.A: target}
    core = Core(table, phases)
    tables = [table]

    assert table.get_target() == target

    factory = Factory(nodes, rails, tables)
    controller = TableAndRailController(table, factory)

    # if we don't move, table is still at start
    res = controller.take_action(Action.none)
    assert res == ActionResult.NONE
    assert start.has_table

    # No valid connection upwards
    res = controller.take_action(Action.up)
    assert res == ActionResult.INVALID

    # Block the way with a table to cause a collision
    block_table = Table(collision)
    res = controller.take_action(Action.down)
    assert res == ActionResult.COLLISION

    # Move table to node_a by first ordering a shuttle from node_b
    res = controller.take_action(Action.right)
    assert table.node == node_a
    assert node_a.has_shuttle
    assert res == ActionResult.MOVED

    # Move table to node_b
    res = controller.take_action(Action.right)
    assert res == ActionResult.MOVED
    assert table.node == node_b
    assert table.has_core()

    # Move table to node_c.
    res = controller.take_action(Action.right)
    assert res == ActionResult.MOVED
    assert table.node == node_c
    assert not table.is_at_target

    # Move table to target. The production phase is over, so the core is gone.
    res = controller.take_action(Action.right)
    assert res == ActionResult.MOVED
    assert table.node == target
    assert not table.has_core()
    assert table.is_at_target
