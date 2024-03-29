from factory.models import Node, Direction, Rail, Table, Core, Phase
from factory.simulation import Factory
import random
import numpy as np

def get_default_factory(random_seed=None, num_tables=8, num_cores=3, num_phases=1,
                        max_num_steps=1000, **kwargs) -> Factory:
    """
                    19--01
                        |
                    00--01--20
                        |
                    16--01      18--03
                    |               |
    02--14--15--15--16--17------17--03
        |                           |
        14--10--09--08--07          |
        |               |           |
        |               07--06--05--03--04---04
        |               |
        14--13--12--11--07
    """
    if random_seed:
        np.random.seed(random_seed)
        random.seed(random_seed)

    node_19   = Node("pt19",   coordinates=(4, 0))
    node_00   = Node("pt00",   coordinates=(4, 1))  # TR
    node_01_c = Node("pt01_c", coordinates=(5, 0))
    node_01_b = Node("pt01_b", coordinates=(5, 1))
    node_01_a = Node("pt01_a", coordinates=(5, 2))
    node_20   = Node("pt20",   coordinates=(6, 1))  # TI
    node_16_b = Node("pt16_b", coordinates=(4, 2))
    node_16_a = Node("pt16_a", coordinates=(4, 3))
    node_02   = Node("pt02",   coordinates=(0, 3))
    node_14_c = Node("pt14_c", coordinates=(1, 3))
    node_14_b = Node("pt14_b", coordinates=(1, 4))
    node_14_a = Node("pt14_a", coordinates=(1, 6))
    node_15_a = Node("pt15_a", coordinates=(2, 3))
    node_15_b = Node("pt15_b", coordinates=(3, 3))
    node_17_a = Node("pt17_a", coordinates=(5, 3))
    node_17_b = Node("pt17_b", coordinates=(7, 3))
    node_18   = Node("pt18",   coordinates=(7, 2))
    node_03_c = Node("pt03_c", coordinates=(8, 2))
    node_03_b = Node("pt03_b", coordinates=(8, 3))
    node_03_a = Node("pt03_a", coordinates=(8, 5))
    node_04_a = Node("pt04_a", coordinates=(9, 5))
    node_04_b = Node("pt04_b", coordinates=(10, 5))
    node_05   = Node("pt05",   coordinates=(7, 5))
    node_06   = Node("pt06",   coordinates=(6, 5))
    node_07_c = Node("pt07_c", coordinates=(5, 4))
    node_07_b = Node("pt07_b", coordinates=(5, 5))
    node_07_a = Node("pt07_a", coordinates=(5, 6))
    node_08   = Node("pt08",   coordinates=(4, 4))
    node_09   = Node("pt09",   coordinates=(3, 4))
    node_10   = Node("pt10",   coordinates=(2, 4))
    node_11   = Node("pt11",   coordinates=(4, 6))
    node_12   = Node("pt12",   coordinates=(3, 6))
    node_13   = Node("pt13",   coordinates=(2, 6))

    node_19.add_neighbour(node_01_a, Direction.right)
    node_00.add_neighbour(node_01_b, Direction.right)
    node_20.add_neighbour(node_01_b, Direction.left)
    node_01_c.add_neighbour(node_01_b, Direction.down)
    node_01_b.add_neighbour(node_01_a, Direction.down)
    node_16_b.add_neighbour(node_01_c, Direction.right)
    node_16_b.add_neighbour(node_16_a, Direction.down)
    node_02.add_neighbour(node_14_c, Direction.right)
    node_14_c.add_neighbour(node_14_b, Direction.down)
    node_14_b.add_neighbour(node_14_a, Direction.down)
    node_14_c.add_neighbour(node_15_a, Direction.right)
    node_15_a.add_neighbour(node_15_b, Direction.right)
    node_15_b.add_neighbour(node_16_a, Direction.right)
    node_16_a.add_neighbour(node_17_a, Direction.right)
    node_17_a.add_neighbour(node_17_b, Direction.right)
    node_17_b.add_neighbour(node_03_b, Direction.right)
    node_18.add_neighbour(node_03_c, Direction.right)
    node_03_c.add_neighbour(node_03_b, Direction.down)
    node_03_b.add_neighbour(node_03_a, Direction.down)
    node_03_a.add_neighbour(node_04_a, Direction.right)
    node_04_a.add_neighbour(node_04_b, Direction.right)
    node_03_a.add_neighbour(node_05, Direction.left)
    node_05.add_neighbour(node_06, Direction.left)
    node_06.add_neighbour(node_07_b, Direction.left)
    node_07_c.add_neighbour(node_07_b, Direction.down)
    node_07_b.add_neighbour(node_07_a, Direction.down)
    node_07_c.add_neighbour(node_08, Direction.left)
    node_08.add_neighbour(node_09, Direction.left)
    node_09.add_neighbour(node_10, Direction.left)
    node_10.add_neighbour(node_14_b, Direction.left)
    node_07_a.add_neighbour(node_11, Direction.left)
    node_11.add_neighbour(node_12, Direction.left)
    node_12.add_neighbour(node_13, Direction.left)
    node_13.add_neighbour(node_14_a, Direction.left)

    nodes = [node_00, node_01_a, node_01_b, node_01_c, node_02, node_03_a, node_03_b, node_03_c,
             node_04_a, node_04_b, node_05, node_06, node_07_a, node_07_b, node_07_c, node_08,
             node_09, node_10, node_11, node_12, node_13, node_14_a, node_14_b, node_14_c,
             node_15_a, node_15_b, node_16_a, node_16_b, node_17_a, node_17_b, node_18, node_19,
             node_20]

    rail_01 = Rail(nodes=[node_01_a, node_01_b, node_01_c])
    rail_16 = Rail(nodes=[node_16_a, node_16_b])
    rail_14 = Rail(nodes=[node_14_a, node_14_b, node_14_c])
    rail_15 = Rail(nodes=[node_15_a, node_15_b])
    rail_17 = Rail(nodes=[node_17_a, node_17_b])
    rail_03 = Rail(nodes=[node_03_a, node_03_b, node_03_c])
    rail_04 = Rail(nodes=[node_04_a, node_04_b])
    rail_07 = Rail(nodes=[node_07_a, node_07_b, node_07_c])

    rails = [rail_01, rail_03, rail_04, rail_07, rail_14, rail_15, rail_16, rail_17]

    tables = create_random_tables_and_cores(nodes, rails, num_tables, num_cores, num_phases)

    return Factory(nodes, rails, tables, max_num_steps, "DefaultFactory")


def get_small_default_factory(random_seed=None, num_tables=4, num_cores=2, num_phases=1,
                              max_num_steps=1000, **kwargs) -> Factory:
    """
    1--2-----2--3
    |           |
    1--6--5     |
    |     |     |
    |     5--4--3
    |     |
    1--7--5
    """
    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)

    node_1_c = Node("pt1_c", coordinates=(0, 0))
    node_1_b = Node("pt1_b", coordinates=(0, 1))
    node_1_a = Node("pt1_a", coordinates=(0, 3))
    node_2_a = Node("pt2_a", coordinates=(1, 0))
    node_2_b = Node("pt2_b", coordinates=(3, 0))
    node_3_b = Node("pt3_b", coordinates=(4, 0))
    node_3_a = Node("pt3_a", coordinates=(4, 2))
    node_4   = Node("pt4",   coordinates=(3, 2))
    node_5_c = Node("pt5_c", coordinates=(2, 1))
    node_5_b = Node("pt5_b", coordinates=(2, 2))
    node_5_a = Node("pt5_a", coordinates=(2, 3))
    node_6   = Node("pt6",   coordinates=(1, 1))
    node_7   = Node("pt7",   coordinates=(1, 3))

    node_1_c.add_neighbour(node_1_b, Direction.down)
    node_1_b.add_neighbour(node_1_a, Direction.down)
    node_1_c.add_neighbour(node_2_a, Direction.right)
    node_2_a.add_neighbour(node_2_b, Direction.right)
    node_2_b.add_neighbour(node_3_b, Direction.right)
    node_3_b.add_neighbour(node_3_a, Direction.down)
    node_3_a.add_neighbour(node_4, Direction.left)
    node_4.add_neighbour(node_5_b, Direction.left)
    node_5_c.add_neighbour(node_5_b, Direction.down)
    node_5_b.add_neighbour(node_5_a, Direction.down)
    node_5_c.add_neighbour(node_6, Direction.left)
    node_5_a.add_neighbour(node_7, Direction.left)
    node_6.add_neighbour(node_1_b, Direction.left)
    node_7.add_neighbour(node_1_a, Direction.left)

    nodes = [node_3_a, node_3_b, node_4, node_5_a, node_5_b, node_5_c, node_6,
             node_7, node_1_a, node_1_b, node_1_c, node_2_a, node_2_b]

    rail_1 = Rail(nodes=[node_1_a, node_1_b, node_1_c])
    rail_2 = Rail(nodes=[node_2_a, node_2_b])
    rail_3 = Rail(nodes=[node_3_a, node_3_b])
    rail_4 = Rail(nodes=[node_5_a, node_5_b, node_5_c])
    rails = [rail_1, rail_2, rail_3, rail_4]

    tables = create_random_tables_and_cores(nodes, rails, num_tables, num_cores, num_phases)

    return Factory(nodes, rails, tables, max_num_steps, "SmallDefaultFactory")


def create_random_tables_and_cores(nodes, rails, num_tables, num_cores, num_phases):
    non_rail_nodes = [n for n in nodes if not n.is_rail]
    for rail in rails:
        # Add precisely one spot on each rail that a table could be placed on
        non_rail_nodes.append(random.choice(rail.nodes))

    tables = []
    table_indices = np.random.choice(range(len(non_rail_nodes)), num_tables, replace=False)
    for i in range(num_tables):
        # Randomly select a node for each table.
        table_idx = random.randint(0, len(non_rail_nodes))
        tables.append(Table(non_rail_nodes[table_indices[i]], name=f"table_{i}"))

    # Core targets go on immobile nodes
    fixed_nodes = [n for n in nodes if not n.is_rail]
    for idx in range(num_cores):
        cycle = {}
        core_indices = np.random.choice(range(len(fixed_nodes)), num_phases, replace=False)
        for p in range(num_phases):
            # For each core phase, randomly select a fixed node.
            cycle[Phase(p)] = fixed_nodes[core_indices[p]]
        Core(tables[idx], cycle, f"core_{idx}")
    return tables
