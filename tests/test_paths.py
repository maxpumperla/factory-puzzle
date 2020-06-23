from factory.util.samples import get_small_default_factory, get_default_factory
from factory.simulation import get_paths_with_distances, get_shortest_path_with_distance

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
    paths_and_distances = get_paths_with_distances(a, b, factory, factor)
    assert len(paths_and_distances) is 3

    paths = [pad[0] for pad in paths_and_distances]
    tables_on_path = [sum([n.has_table()for n in path]) for path in paths]
    assert tables_on_path == [1, 2, 2]

    assert [pad[1] for pad in paths_and_distances] == [4 * 1 + 1 * factor, 5 * 1 + 2 * factor , 6 * 1 + 2 * factor]

    shortest = get_shortest_path_with_distance(a, b, factory, factor)
    assert shortest[1] is 4 * 1 + 1 * factor
