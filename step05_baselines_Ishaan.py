"""Step 5 - Ishaan: UCS and Greedy Best-First Search.

UCS is precisely A* with h=0, so it deliberately reuses the reviewed engine.
Greedy is separate: priority ignores g, and a node is discovered only once.
"""
from step02_heap_Rohan import MinHeap
from step03_heuristics_Rahul import zero, euclidean
from step04_astar_Nimisha import astar, reconstruct_path


def uniform_cost(graph, start, goal):
    return astar(graph, start, goal, zero)


def greedy(graph, start, goal, heuristic=euclidean):
    if start not in graph["nodes"] or goal not in graph["nodes"]:
        raise ValueError("Start and goal must be graph node IDs.")
    frontier = MinHeap()
    sequence = 0
    frontier.push((heuristic(graph, start, goal), sequence, start, 0.0))
    discovered = {start}
    parent, costs = {}, {start: 0.0}
    order, trace = [], []
    while frontier:
        priority, _, node, _ = frontier.pop()
        trace.append({"node": node, "g": costs[node], "h": priority, "f": priority})
        if node == goal:
            return {"path": reconstruct_path(parent, goal), "cost": costs[node],
                    "expanded": len(order), "order": order, "trace": trace, "reopened": 0}
        order.append(node)
        for neighbor, edge_cost in graph["edges"][node].items():
            if neighbor not in discovered:
                discovered.add(neighbor)
                parent[neighbor] = node
                costs[neighbor] = costs[node] + edge_cost
                sequence += 1
                frontier.push((heuristic(graph, neighbor, goal), sequence,
                               neighbor, costs[neighbor]))
    return {"path": [], "cost": None, "expanded": len(order), "order": order,
            "trace": trace, "reopened": 0}


if __name__ == "__main__":
    from step01_graph_Tanishq import build_graph
    graph = build_graph()
    for name, algorithm in [("UCS", uniform_cost), ("Greedy", greedy)]:
        result = algorithm(graph, "MAIN_GATE", "SJT")
        print(name, result["path"], round(result["cost"], 3), result["expanded"])
