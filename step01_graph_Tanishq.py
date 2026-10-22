"""Step 1 - Tanishq: a small, manually traced campus walking graph.

Coordinates refer to a 1882 x 1344 rendering of Campus-Map.jpg. Points are
approximate ROAD ACCESS points, not building centres. There is no map scale:
one reference pixel is one MODEL metre by default, not a surveyed metre.
Edit this dataset after walking the paths. The AI never infers roads from pixels.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE_SIZE = (1882, 1344)

# id: (display name, x, y, selectable landmark)
POINTS = {
    "MAIN_GATE": ("Main Gate", 436, 1090, True),
    "MGR": ("Main Block (MGR) access", 450, 1015, True),
    "WEST_CENTER": ("West central junction", 505, 1015, False),
    "LIBRARY": ("Library access", 580, 1015, True),
    "SMV": ("SMV access", 740, 1024, True),
    "CBMR": ("CBMR access", 851, 1040, True),
    "LH_AB": ("Ladies Hostel A/B access", 740, 1127, True),
    "GATE3": ("Gate 3", 740, 1180, True),
    "WEST_MID": ("West middle junction", 505, 913, False),
    "LIB_N": ("Library north junction", 580, 910, False),
    "SMV_N": ("SMV north junction", 739, 913, False),
    "ANNA": ("Anna Auditorium access", 368, 913, True),
    "GDN": ("GDN access", 365, 841, True),
    "GATE2A": ("Gate 2A", 237, 818, True),
    "WOODYS_N": ("Woodys north junction", 520, 767, False),
    "WOODYS_S": ("Woodys Parking access", 688, 911, True),
    "S3_S": ("Subway S3 south", 712, 764, False),
    "S3_N": ("Subway S3 north", 710, 655, False),
    "MH_WEST": ("Men's hostel west junction", 517, 652, False),
    "MH_HJ": ("Men's Hostel H/J access", 517, 558, True),
    "MH_AB": ("Men's Hostel A/B access", 517, 439, True),
    "MH_B": ("Men's Hostel B north access", 517, 307, True),
    "MH_F": ("Men's Hostel F access", 635, 439, True),
    "MH_G": ("Men's Hostel G access", 845, 433, True),
    "MH_CENTER": ("Men's hostel centre junction", 731, 588, False),
    "MH_OFFICE": ("Men's Hostel Office access", 794, 588, True),
    "MH_EAST": ("Men's hostel east junction", 933, 583, False),
    "MH_K": ("Men's Hostel K access", 1005, 563, True),
    "MH_L": ("Men's Hostel L access", 1229, 526, True),
    "MH_M": ("Men's Hostel M access", 1390, 518, True),
    "MH_PR": ("Men's Hostel P/R access", 1390, 448, True),
    "MH_N": ("Men's Hostel N access", 1400, 336, True),
    "NORTH_E": ("North east junction", 1598, 420, False),
    "S1_N": ("Subway S1 north", 1470, 430, False),
    "S1_S": ("Subway S1 south", 1487, 618, False),
    "S2_N": ("Subway S2 north", 933, 660, False),
    "S2_S": ("Subway S2 south", 937, 731, False),
    "TT_N": ("Technology Tower north junction", 750, 761, False),
    "TT": ("Technology Tower access", 748, 840, True),
    "LAKE_NW": ("Lake north west junction", 904, 731, False),
    "LH_C": ("Ladies Hostel C/D access", 1036, 714, True),
    "LH_OFFICE": ("Ladies Hostel Office access", 1230, 668, True),
    "SJT_J": ("SJT north junction", 1313, 658, False),
    "SJT": ("SJT access", 1390, 817, True),
    "PRP": ("PRP access", 1550, 812, True),
    "PRP_SE": ("PRP south east junction", 1690, 800, False),
    "EAST_N": ("East road junction", 1741, 547, False),
    "GATE5": ("Gate 5", 1805, 580, True),
    "MGB": ("MGB access", 1741, 665, True),
    "PARK_E": ("East parking junction", 1596, 984, False),
    "SOUTH_E": ("South east junction", 1363, 975, False),
    "GATE11": ("Gate 11", 1324, 1295, True),
}

# (from, to, intermediate road bends). Every road below allows both directions.
# Two directed arcs represent each road. No edge crosses the lake or railway
# except the four explicitly modelled pedestrian subway links S1-S4.
ROADS = [
    ("MAIN_GATE", "MGR", []), ("MGR", "WEST_CENTER", []),
    ("WEST_CENTER", "LIBRARY", []), ("LIBRARY", "SMV", []),
    ("SMV", "CBMR", []), ("SMV", "LH_AB", []), ("LH_AB", "GATE3", []),
    ("WEST_CENTER", "WEST_MID", []), ("WEST_MID", "LIB_N", []),
    ("LIB_N", "SMV_N", []), ("LIBRARY", "LIB_N", []),
    ("SMV", "SMV_N", []), ("WEST_MID", "ANNA", []),
    ("ANNA", "GDN", []), ("GDN", "GATE2A", [(365, 817)]),
    ("WEST_MID", "WOODYS_N", []), ("WEST_MID", "WOODYS_S", []),
    ("WOODYS_S", "S3_S", []), ("WOODYS_N", "S3_S", []),
    ("WOODYS_N", "MH_WEST", [(515, 719)]),  # Subway S4
    ("S3_S", "S3_N", []),  # Subway S3
    ("MH_WEST", "S3_N", [(691, 652)]),
    ("MH_WEST", "MH_HJ", []), ("MH_HJ", "MH_AB", []),
    ("MH_AB", "MH_B", []), ("MH_AB", "MH_F", []),
    ("MH_F", "MH_G", []), ("MH_G", "MH_EAST", [(1006, 425), (1008, 560)]),
    ("MH_HJ", "MH_CENTER", [(674, 551)]),
    ("S3_N", "MH_CENTER", [(722, 649)]),
    ("MH_CENTER", "MH_OFFICE", []), ("MH_OFFICE", "MH_EAST", []),
    ("MH_EAST", "MH_K", []), ("MH_K", "MH_L", []),
    ("MH_L", "MH_M", []), ("MH_M", "MH_PR", []),
    ("MH_PR", "MH_N", []), ("MH_N", "NORTH_E", [(1505, 336), (1598, 343)]),
    ("NORTH_E", "S1_N", []), ("MH_M", "S1_N", [(1390, 430)]),
    ("S1_N", "S1_S", [(1462, 488)]),  # Subway S1
    ("MH_EAST", "S2_N", []), ("S2_N", "S2_S", []),  # Subway S2
    ("S3_S", "TT_N", [(744, 749)]), ("TT_N", "TT", []),
    ("SMV_N", "TT", []), ("TT_N", "LAKE_NW", [(850, 737)]),
    ("S2_S", "LAKE_NW", []), ("S2_S", "LH_C", []),
    ("LH_C", "LH_OFFICE", [(1150, 694)]),
    ("LH_OFFICE", "SJT_J", []), ("SJT_J", "S1_S", []),
    ("SJT_J", "SJT", [(1393, 725)]), ("S1_S", "PRP", [(1505, 743)]),
    ("PRP", "PRP_SE", []), ("S1_S", "EAST_N", []),
    ("EAST_N", "GATE5", []), ("EAST_N", "MGB", []),
    ("MGB", "PRP_SE", [(1735, 675), (1696, 675)]),
    ("SJT", "SOUTH_E", []),
    ("PRP_SE", "PARK_E", [(1685, 904), (1578, 908)]),
    ("PARK_E", "SOUTH_E", []), ("SOUTH_E", "GATE11", []),
]


def validate_graph(graph):
    """Reject malformed data before a search (positive finite edge costs)."""
    nodes, edges = graph["nodes"], graph["edges"]
    if not nodes or not math.isfinite(graph["scale"]) or graph["scale"] <= 0:
        raise ValueError("The graph needs nodes and a positive finite scale.")
    if set(edges) != set(nodes):
        raise ValueError("Every node needs an adjacency list, including isolated nodes.")
    for node, record in nodes.items():
        if not all(math.isfinite(record[k]) for k in ("x", "y")):
            raise ValueError(f"Invalid coordinates: {node}")
        for neighbor, cost in edges[node].items():
            if neighbor not in nodes or not math.isfinite(cost) or cost <= 0:
                raise ValueError(f"Invalid edge: {node}:{neighbor}")
            a, b = record, nodes[neighbor]
            lower_bound = math.hypot(a["x"] - b["x"], a["y"] - b["y"]) * graph["scale"]
            if cost + 1e-9 < lower_bound:
                raise ValueError(f"Edge shorter than straight line: {node}:{neighbor}")
    return graph


def build_graph(scale=1.0, closed=()):
    """Generate the dataset. A closure removes BOTH directions of a road."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be a positive finite number.")
    nodes = {key: {"name": v[0], "x": v[1], "y": v[2], "landmark": v[3]}
             for key, v in POINTS.items()}
    graph = {"nodes": nodes, "edges": {key: {} for key in nodes},
             "shapes": {}, "scale": scale, "units": "estimated model metres",
             "reference_size": list(REFERENCE_SIZE)}
    for a, b, bends in ROADS:
        shape = [(nodes[a]["x"], nodes[a]["y"]), *bends, (nodes[b]["x"], nodes[b]["y"])]
        length = sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(shape, shape[1:]))
        # Round upwards so rounding cannot break the Euclidean lower bound.
        cost = math.ceil(length * scale * 1000) / 1000
        graph["edges"][a][b] = graph["edges"][b][a] = cost
        graph["shapes"][f"{a}:{b}"] = shape
        graph["shapes"][f"{b}:{a}"] = list(reversed(shape))
    for road in closed:
        pieces = road.upper().split(":")
        if len(pieces) != 2:
            raise ValueError("A closure must look like S3_S:S3_N.")
        a, b = pieces
        if a not in nodes or b not in graph["edges"][a]:
            raise ValueError(f"Unknown or duplicate closure: {road}")
        del graph["edges"][a][b]
        graph["edges"][b].pop(a, None)
        graph["shapes"].pop(f"{a}:{b}", None)
        graph["shapes"].pop(f"{b}:{a}", None)
    return validate_graph(graph)


def save_graph(graph, destination=None):
    destination = Path(destination or ROOT / "output" / "campus_graph.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return destination


if __name__ == "__main__":
    graph = build_graph()
    print(f"Created {len(graph['nodes'])} nodes and {sum(map(len, graph['edges'].values()))} directed arcs.")
    print(save_graph(graph))
    print("Distances are estimates. Map connections need an on-campus check.")
