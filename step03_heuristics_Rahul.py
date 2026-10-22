"""Step 3 - Rahul: distance estimates and the conditions behind them.

All costs and heuristic values use the SAME scale. For each model edge,
cost >= Euclidean displacement. Triangle inequality then makes Euclidean
consistent. (|dx|+|dy|)/sqrt(2) <= Euclidean, and scaled L1 is a metric,
so scaled Manhattan is also consistent. Raw L1 has no such guarantee.
"""
import math


def zero(graph, node, goal):
    return 0.0


def euclidean(graph, node, goal):
    a, b = graph["nodes"][node], graph["nodes"][goal]
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"]) * graph["scale"]


def manhattan(graph, node, goal):
    a, b = graph["nodes"][node], graph["nodes"][goal]
    return (abs(a["x"] - b["x"]) + abs(a["y"] - b["y"])) * graph["scale"]


def scaled_manhattan(graph, node, goal):
    return manhattan(graph, node, goal) / math.sqrt(2)


HEURISTICS = {"zero": zero, "euclidean": euclidean,
              "manhattan": manhattan, "scaled_manhattan": scaled_manhattan}


def check_consistency(graph, heuristic):
    """Exhaustively check h(u) <= c(u,v)+h(v) for all goals and arcs."""
    failures = []
    for goal in graph["nodes"]:
        for a, neighbors in graph["edges"].items():
            for b, cost in neighbors.items():
                if heuristic(graph, a, goal) > cost + heuristic(graph, b, goal) + 1e-8:
                    failures.append((a, b, goal))
    return failures


if __name__ == "__main__":
    from step01_graph_Tanishq import build_graph
    graph = build_graph()
    for name, function in HEURISTICS.items():
        print(name, "consistency violations:", len(check_consistency(graph, function)))
