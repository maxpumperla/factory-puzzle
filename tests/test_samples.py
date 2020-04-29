from factory.util.samples import get_default_factory
from factory.util.writer import print_factory


def test_default_factory():
    factory = get_default_factory(random_seed=1337)
    assert len(factory.nodes) == 33
    assert len(factory.rails) == 8
    assert len(factory.tables) == 8
    assert factory.name == "DefaultFactory"

    assert factory.tables[0].node.coordinates == (1, 6)
    assert factory.tables[0].core.current_target.coordinates == (4, 1)

    # print_factory(factory)
