from ..models import Factory, Node, Direction
SCALE_X, SCALE_Y = 4, 2


def factory_string(factory: Factory) -> str:
    nodes = factory.nodes
    max_x = max([n.coordinates[0] for n in nodes]) + 1
    max_y = max([n.coordinates[1] for n in nodes]) + 1

    grid = [[" "] * max_x * SCALE_X for _ in range(max_y * SCALE_Y)]

    for node in nodes:
        x, y = node.coordinates
        text = node_text(node)
        grid[y * SCALE_Y][x * SCALE_X] = text
        for direction, nb in node.neighbours.items():
            if nb:
                if direction == "left":
                    grid[y * SCALE_Y][x * SCALE_X - SCALE_X + 1: x * SCALE_X] = ["-"] * (SCALE_X - 1)
                elif direction == "right":
                    grid[y * SCALE_Y][x * SCALE_X + 1: x * SCALE_X + SCALE_X] = ["-"] * (SCALE_X - 1)
                elif direction == "up":
                    grid[y * SCALE_Y - 1][x * SCALE_X] = "|"
                elif direction == "down":
                    grid[y * SCALE_Y + 1][x * SCALE_X] = "|"
    grid = ["".join(line) for line in grid]
    return "\n".join(grid)


def print_factory(factory: Factory):
    clear_screen()
    print(factory_string(factory))


def node_text(node: Node) -> str:
    text = "N"
    if node.has_table():
        text = "T"
        if node.table.has_core():
            text = "C"
    return text


def clear_screen():
    print(chr(27) + "[2J")
