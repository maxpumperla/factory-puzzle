from ..models import Factory, Node, Direction
import cv2
import numpy as np
SCALE_X, SCALE_Y = 4, 2


def draw_boxes(factory: Factory, original_img: np.array):
    img = np.copy(original_img)
    for node in factory.nodes:
        coords = node.coordinates
        pos = (coords[0] * 100, coords[1] * 100)
        text = node_text(node)
        img = draw_box(img, pos, text)
    return img


def draw_box(img: np.array, pos=(0, 0), text: str = "C"):
    top_left = pos
    bottom_right = (pos[0] + 100, pos[1] + 100)
    if text is "C":
        thickness = -1
        color = (255, 0, 0)
    elif text is "T":
        thickness = -1
        color = (0, 255, 0)
    else:
        thickness = 4
        color = (255, 0, 0)

    img = cv2.rectangle(img, top_left, bottom_right, color, thickness)

    position = (pos[0] + 45, pos[1] + 50)
    if text in ["C", "T"]:
        img = cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
    return img


def factory_string(factory: Factory, fill_char="·", line_break="\n") -> str:
    nodes = factory.nodes
    max_x = max([n.coordinates[0] for n in nodes]) + 1
    max_y = max([n.coordinates[1] for n in nodes]) + 1

    grid = [[fill_char] * max_x * SCALE_X for _ in range(max_y * SCALE_Y)]

    for node in nodes:
        x, y = node.coordinates
        text = node_text(node)
        grid[y * SCALE_Y][x * SCALE_X] = text
        for direction, nb in node.neighbours.items():
            if nb:
                if direction == "left":
                    grid[y * SCALE_Y][x * SCALE_X - SCALE_X + 1: x * SCALE_X] = ["="] * (SCALE_X - 1)
                elif direction == "right":
                    grid[y * SCALE_Y][x * SCALE_X + 1: x * SCALE_X + SCALE_X] = ["="] * (SCALE_X - 1)
                elif direction == "up":
                    grid[y * SCALE_Y - 1][x * SCALE_X] = "║"
                elif direction == "down":
                    grid[y * SCALE_Y + 1][x * SCALE_X] = "║"
    grid = ["".join(line) for line in grid]
    return line_break.join(grid)


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
