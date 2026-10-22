"""Step 4 - Nimisha: A* graph search, implemented without an AI library."""
from step02_heap_Rohan import MinHeap
from step03_heuristics_Rahul import euclidean


def reconstruct_path(parent, goal):
    path = [goal]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    return list(reversed(path))


def astar(graph, start, goal, heuristic=euclidean):
    if start not in graph["nodes"] or goal not in graph["nodes"]:
        raise ValueError("Start and goal must be graph node IDs.")
    frontier = MinHeap()
    sequence = 0
    frontier.push((heuristic(graph, start, goal), sequence, start, 0.0))
    best_g = {start: 0.0}
    parent = {}
    expanded_order = []
    trace = []
    expanded_cost = {}
    reopened = 0
    while frontier:
        f, _, node, popped_g = frontier.pop()
        # Lazy decrease-key: an improved route inserts a NEW heap entry.
        # Skip the old entry when it is popped. It is not a new expansion.
        if popped_g > best_g[node]:
            continue
        if node in expanded_cost:
            if popped_g >= expanded_cost[node]:
                continue
            reopened += 1
        # Count an expansion only when we generate successors. The goal pop
        # is recorded in the trace but is not counted as an expansion.
        trace.append({"node": node, "g": popped_g,
                      "h": heuristic(graph, node, goal), "f": f})
        if node == goal:
            return {"path": reconstruct_path(parent, goal), "cost": popped_g,
                    "expanded": len(expanded_order), "order": expanded_order,
                    "trace": trace, "reopened": reopened}
        expanded_cost[node] = popped_g
        expanded_order.append(node)
        for neighbor, edge_cost in graph["edges"][node].items():
            candidate = popped_g + edge_cost
            # This is the AI's route-improvement decision. Reopen a previously
            # expanded node if a cheaper route reaches it. This matters for
            # admissible but inconsistent heuristics on general graphs.
            if candidate < best_g.get(neighbor, float("inf")):
                best_g[neighbor] = candidate
                parent[neighbor] = node
                sequence += 1
                priority = candidate + heuristic(graph, neighbor, goal)
                frontier.push((priority, sequence, neighbor, candidate))
    return {"path": [], "cost": None, "expanded": len(expanded_order),
            "order": expanded_order, "trace": trace, "reopened": reopened}


if __name__ == "__main__":
    from step01_graph_Tanishq import build_graph
    result = astar(build_graph(), "MAIN_GATE", "SJT")
    print("A* route:", " -> ".join(result["path"]))
    print("Estimated model metres:", round(result["cost"], 3))
    print("Expansions:", result["expanded"])
