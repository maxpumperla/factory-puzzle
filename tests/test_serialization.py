import json

from factory.models import Node, Table, Core, Phase, Direction
from factory.serialization import node_to_dict, table_to_dict, core_to_dict


def test_serde():
    node = Node(name="foo", is_rail=False, coordinates=(0,0))
    node_b = Node(coordinates=(1,0))
    node.add_neighbour(node_b, Direction.right)
    assert json.dumps(node_to_dict(node)) == '{"name": "foo", "neighbours": {"right": [1, 0]}, "is_rail": false, "has_shuttle": true, "coordinates": [0, 0]}'

    table = Table(node=node, core=None, name="bar")
    assert json.dumps(table_to_dict(table)) == '{"name": "bar", "coordinates": [0, 0]}'

    phases = {Phase.A: node}
    core = Core(table=table, cycle=phases, name="baz")
    assert json.dumps(core_to_dict(core)) == '{"name": "baz", "coordinates": [0, 0], "current_phase": "A", "current_target_coordinates": [0, 0]}'
