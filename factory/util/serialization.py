"""To track movement and visualize a simulation we only need names,
positional information and so on."""
from factory.models import *
import json


def core_to_dict(core: Core) -> dict:
    return {
        "name": core.name,
        "coordinates": core.table.node.coordinates,
        "current_phase": core.current_phase.name,
        "current_target_coordinates": core.current_target.coordinates,
    }


def table_to_dict(table: Table) -> dict:
    return {
        "name": table.name,
        "coordinates": table.node.coordinates
    }


def node_to_dict(node: Node) -> dict:
    return {
        "name": node.name,
        "neighbours": {k:v.coordinates for k,v in node.neighbours.items() if v},
        "is_rail": node.is_rail,
        "has_shuttle": node.has_shuttle,
        "coordinates": node.coordinates,
    }
