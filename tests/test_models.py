import pytest
from factory.models import Direction, Node, Rail, Table, Core, Phase
from factory.simulation import Factory


def test_directions():
    up = Direction(0)
    down = up.opposite()
    assert down.name == 'down'
    assert down.opposite() == up
    right = Direction(1)
    left = right.opposite()
    assert left.value == 3
    assert left.opposite() == right


def test_get_all():
    assert Direction.get_all() == ['up', 'right', 'down', 'left']


def test_node_connectivity():
    node_a = Node()
    # nodes are not connected when created
    assert all(not node_a.connected_to(d) for d in Direction.get_all())

    with pytest.raises(Exception):
        # no self-connections
        node_a.add_neighbour(node_a, Direction.left)

    node_b = Node()
    # b <- a
    node_a.add_neighbour(node_b, Direction.left)

    # nodes are mutually connected now
    assert node_a.connected_to(node_b)
    assert node_b.connected_to(node_a)

    assert node_a.has_neighbour(Direction.left)
    assert node_b.has_neighbour(Direction.right)

    node_c = Node()
    with pytest.raises(Exception):
        # a already has a left neighbour
        node_a.add_neighbour(node_c, Direction.left)

    # connections have a direction we can test
    assert node_a.has_neighbour(Direction(3))
    assert node_b.has_neighbour(Direction.right)


def test_rail():
    node_a, node_b, node_c = Node(), Node(), Node()
    node_a.add_neighbour(node_b, Direction.left)
    node_b.add_neighbour(node_c, Direction.up)

    rail = Rail(nodes=[node_a, node_b, node_c], shuttle=node_a)

    # the rail has a shuttle at a
    assert node_a.has_shuttle
    assert not node_a.has_table()
    assert rail.shuttle == node_a

    # shuttles can be moved along rails
    rail.order_shuttle(to=node_c)
    assert rail.shuttle == node_c
    assert not node_a.has_shuttle

    # put a table on the shuttle
    table = Table(node_c)
    assert node_c.table == table
    assert table.node == node_c

    with pytest.raises(Exception):
        # Can't order shuttles with tables on them
        rail.order_shuttle(to=node_b)


def test_get_rail():
    node_a, node_b, node_c = Node(), Node(), Node()
    node_a.add_neighbour(node_b, Direction.left)
    node_b.add_neighbour(node_c, Direction.up)

    nodes = [node_a, node_b, node_c]
    rail = Rail(nodes=nodes, shuttle=node_a)
    factory = Factory(nodes, [rail], [])

    assert factory.get_rail(node_a) == rail

    node_d = Node()
    assert factory.get_rail(node_d) is None


def test_core():
    node = Node()
    target = Node()
    phases = {Phase.A: target}
    table = Table(node)
    core = Core(table, phases)

    assert table.has_core
    assert table.core == core
    assert core.table == table
    assert core.current_phase == Phase.A
    assert core.current_target == target
