from factory.util.samples import get_default_factory, get_small_default_factory


def test_default_factory():
    factory = get_default_factory(random_seed=1337)
    assert len(factory.nodes) == 33
    assert len(factory.rails) == 8
    assert len(factory.tables) == 8
    assert factory.name == "DefaultFactory"

    assert factory.tables[0].node.coordinates == (4, 0)
    assert factory.tables[0].core.current_target.coordinates == (6, 1)


def test_no_rails_default_factory():
    factory = get_default_factory(random_seed=1337, with_rails=False)
    assert len(factory.nodes) == 33
    assert len(factory.rails) == 0
    assert len(factory.tables) == 8
    assert factory.name == "DefaultFactory"

    assert sum([n.is_rail for n in factory.nodes]) == 0

    assert factory.tables[0].node.coordinates == (2, 3)
    assert factory.tables[0].core.current_target.coordinates == (8, 5)


def test_small_factory():
    factory = get_small_default_factory(random_seed=1337)
    assert len(factory.nodes) == 13

    assert factory.name == "SmallDefaultFactory"

    assert factory.tables[0].node.coordinates == (2, 2)
    assert factory.tables[0].core.current_target.coordinates == (1, 1)
