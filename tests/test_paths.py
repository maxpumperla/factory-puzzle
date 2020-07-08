from factory.util.samples import get_small_default_factory, get_default_factory
from factory.simulation import get_shortest_weighted_path, get_paths_distances_obstructions
from factory.heuristics import move_table_along_path


def test_small_factory_paths():
    factory = get_small_default_factory()
    assert factory.graph.number_of_nodes() is 13
    assert factory.graph.number_of_edges() is 14

    nodes = factory.nodes
    a = nodes[0]  # pt3_a
    b = nodes[10]  # pt1_c
    a_to_b_paths = factory.get_paths(a, b)

    assert len(a_to_b_paths) is 3
    assert sorted(len(p) for p in a_to_b_paths) == [5, 7, 8]

    assert len(factory.paths) is 1
    assert (0, 10) in factory.paths.keys()


def test_big_factory_paths():
    factory = get_default_factory()
    assert factory.graph.number_of_nodes() is 33
    assert factory.graph.number_of_edges() is 34

    nodes = factory.nodes
    a = nodes[5]  # pt3_a
    b = nodes[23]  # pt14_c
    a_to_b_paths = factory.get_paths(a, b)

    assert len(a_to_b_paths) is 3
    assert sorted(len(p) for p in a_to_b_paths) == [8, 10, 11]


def test_get_paths_with_distances():
    factory = get_small_default_factory(random_seed=1337)

    nodes = factory.nodes
    a = nodes[0]  # pt3_a
    b = nodes[10]  # pt1_c

    factor = 4
    paths_and_distances = get_paths_distances_obstructions(a, b, factory, factor)
    assert len(paths_and_distances) is 3

    paths = [pad[0] for pad in paths_and_distances]
    tables_on_path = [sum([n.has_table()for n in path]) for path in paths]
    assert tables_on_path == [1, 3, 3]

    assert [pad[1] for pad in paths_and_distances] == [4 * 1 + 1 * factor, 4 * 1 + 3 * factor , 5 * 1 + 3 * factor]

    shortest = get_shortest_weighted_path(a, b, factory, factor)
    assert shortest[1] is 4 * 1 + 1 * factor


def test_small_unobstructed_paths():
    factory = get_small_default_factory(num_tables=1, num_cores=1)

    nodes = factory.nodes
    a = nodes[5]  # pt5_c
    b = nodes[0]  # pt3_a
    a_to_b_paths = factory.get_paths(a, b)
    assert len(a_to_b_paths) is 4

    unobstructed_paths = factory.get_unobstructed_paths(a, b)
    assert len(unobstructed_paths) is 2

    # paths, weighted_distances, distances, obstructions, tables_numbers
    pdo = get_paths_distances_obstructions(a, b, factory)
    assert len(pdo) is 4
    assert pdo[0][1] == 4
    assert pdo[0][2] == 4
    assert pdo[0][3] == []
    assert pdo[0][4] == 0


def test_move_along_path():
    factory = get_small_default_factory(num_tables=1, num_cores=1)

    nodes = factory.nodes
    target = nodes[0]  # pt3_a

    table = factory.tables[0]
    node = table.node
    assert not target.has_table()

    paths = factory.get_paths(node, target)
    path = paths[0]

    move_table_along_path(path, factory)
    assert target.has_table()



