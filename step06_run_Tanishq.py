"""Step 6 - Tanishq: command-line demo, comparison tables, and verification.

The whole project uses only the Python standard library. Run `python step06_run_Tanishq.py --help` for all options.
"""
import argparse
import csv
import json
import math
import platform
import random
import statistics
import sys
from pathlib import Path
from time import perf_counter_ns

from step01_graph_Tanishq import ROOT, build_graph, save_graph, validate_graph
from step02_heap_Rohan import MinHeap
from step03_heuristics_Rahul import (HEURISTICS, check_consistency, euclidean,
                                     manhattan, scaled_manhattan, zero)
from step04_astar_Nimisha import astar
from step05_baselines_Ishaan import greedy, uniform_cost

ALGORITHMS = {
    "UCS": uniform_cost,
    "Greedy Euclidean": greedy,
    "A* Euclidean": lambda g, s, t: astar(g, s, t, euclidean),
    "A* Scaled Manhattan": lambda g, s, t: astar(g, s, t, scaled_manhattan),
    "A* Raw Manhattan": lambda g, s, t: astar(g, s, t, manhattan),
}
GUARANTEED = {"UCS", "A* Euclidean", "A* Scaled Manhattan"}


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_distances(graph):
    """Independent Floyd-Warshall oracle, only for verification, not routing."""
    nodes = list(graph["nodes"])
    distances = {a: {b: (0.0 if a == b else graph["edges"][a].get(b, math.inf))
                     for b in nodes} for a in nodes}
    for middle in nodes:
        for a in nodes:
            for b in nodes:
                distances[a][b] = min(distances[a][b],
                                       distances[a][middle] + distances[middle][b])
    return distances


def validate_result(graph, start, goal, result, expected=None):
    """A legal returned path must agree with its own accumulated edge costs."""
    path = result["path"]
    if not path:
        assert result["cost"] is None
        if expected is not None:
            assert math.isinf(expected)
        return
    assert path[0] == start and path[-1] == goal
    assert len(path) == len(set(path)), "Positive-cost shortest paths have no cycles"
    cost = sum(graph["edges"][a][b] for a, b in zip(path, path[1:]))
    assert math.isclose(cost, result["cost"], abs_tol=1e-7)
    if expected is not None:
        assert math.isclose(cost, expected, abs_tol=1e-7), (start, goal, cost, expected)


def toy_graph(coordinates, arcs):
    graph = {"nodes": {n: {"name": n, "x": xy[0], "y": xy[1], "landmark": True}
                       for n, xy in coordinates.items()},
             "edges": {n: {} for n in coordinates}, "scale": 1.0}
    for a, b, cost in arcs:
        graph["edges"][a][b] = cost
    return validate_graph(graph)


def verify(graph):
    """Checks cover the algorithm's guarantees, not just a rehearsed route."""
    rng = random.Random(2026)
    heap = MinHeap()
    values = [(rng.randint(-10, 10), i) for i in range(500)]
    for value in values:
        heap.push(value)
    assert [heap.pop() for _ in values] == sorted(values)
    try:
        heap.pop()
        raise AssertionError("Empty heap must raise IndexError")
    except IndexError:
        pass
    # Mixed insertion/removal catches bugs that a single sorted drain misses.
    bag = []
    for i in range(500):
        if bag and rng.random() < 0.5:
            expected = min(bag)
            bag.remove(expected)
            assert heap.pop() == expected
        else:
            value = (rng.randint(-10, 10), i)
            heap.push(value)
            bag.append(value)
    assert [heap.pop() for _ in bag] == sorted(bag)

    oracle = exact_distances(graph)
    comparisons = 0
    raw_suboptimal = 0
    greedy_suboptimal = 0
    for start in graph["nodes"]:
        for goal in graph["nodes"]:
            for name, algorithm in ALGORITHMS.items():
                result = algorithm(graph, start, goal)
                validate_result(graph, start, goal, result,
                                oracle[start][goal] if name in GUARANTEED else None)
                if name == "A* Raw Manhattan" and result["path"]:
                    raw_suboptimal += result["cost"] > oracle[start][goal] + 1e-7
                if name == "Greedy Euclidean" and result["path"]:
                    greedy_suboptimal += result["cost"] > oracle[start][goal] + 1e-7
                comparisons += 1

    audit = {}
    for name, function in HEURISTICS.items():
        violations = check_consistency(graph, function)
        overestimates = sum(function(graph, node, goal) > oracle[node][goal] + 1e-8
                            for node in graph["nodes"] for goal in graph["nodes"])
        audit[name] = {"consistency_violations": len(violations),
                       "admissibility_violations": overestimates,
                       "example_inconsistent_arc_and_goal": list(violations[0]) if violations else None}
        if name != "manhattan":
            assert not violations and overestimates == 0

    # An admissible but inconsistent heuristic forces A to reopen.
    special = toy_graph({"S": (0, 0), "A": (0, 0), "B": (0, 0), "G": (0, 0), "U": (0, 0)},
                        [("S", "A", 3), ("S", "B", 1), ("B", "A", 1), ("A", "G", 2)])
    estimates = {"S": 0, "A": 0, "B": 3, "G": 0, "U": 0}
    result = astar(special, "S", "G", lambda g, n, t: estimates[n])
    assert result["cost"] == 4 and result["reopened"] == 1
    for algorithm in ALGORITHMS.values():
        assert algorithm(special, "G", "S")["path"] == []  # directed graph
        assert algorithm(special, "S", "U")["cost"] is None
        assert algorithm(special, "S", "S")["cost"] == 0
        try:
            algorithm(special, "MISSING", "G")
            raise AssertionError("Invalid node must be rejected")
        except ValueError:
            pass

    diagonal = toy_graph({"S": (0, 0), "A": (1, 1), "G": (2, 2)},
                         [("S", "G", 3), ("S", "A", 1.415), ("A", "G", 1.415)])
    assert math.isclose(astar(diagonal, "S", "G", euclidean)["cost"], 2.83)
    assert astar(diagonal, "S", "G", manhattan)["cost"] == 3
    # Independent random directed graphs cover dead ends and stale heap entries.
    random_cases = 0
    for _ in range(20):
        names = [str(i) for i in range(7)]
        arcs = [(a, b, rng.randint(1, 20)) for a in names for b in names
                if a != b and rng.random() < 0.25]
        sample = toy_graph({n: (0, 0) for n in names}, arcs)
        expected = exact_distances(sample)
        for a in names:
            for b in names:
                validate_result(sample, a, b, uniform_cost(sample, a, b), expected[a][b])
                random_cases += 1
    closed = build_graph(graph["scale"], ["S3_S:S3_N"])
    assert "S3_N" not in closed["edges"]["S3_S"]
    validate_result(closed, "MAIN_GATE", "MH_N", astar(closed, "MAIN_GATE", "MH_N"),
                    exact_distances(closed)["MAIN_GATE"]["MH_N"])
    for scale in (0, -1, math.inf, math.nan):
        try:
            build_graph(scale)
            raise AssertionError("Invalid scale must be rejected")
        except ValueError:
            pass
    summary = {"passed": True, "campus_route_checks": comparisons,
               "random_directed_oracle_checks": random_cases,
               "campus_raw_manhattan_suboptimal_pairs": raw_suboptimal,
               "campus_greedy_suboptimal_pairs": greedy_suboptimal,
               "heuristic_audit": audit,
               "special_checks": ["heap", "same node", "unreachable", "directed edges",
                                  "invalid input", "reopening", "diagonal overestimate", "closure"]}
    print(f"Verification passed: {comparisons} campus searches and {random_cases} random directed checks.")
    return summary


def benchmark(graph, output, count=50, repeats=21, seed=42):
    landmarks = [n for n, v in graph["nodes"].items() if v["landmark"]]
    pairs = [(s, t) for s in landmarks for t in landmarks if s != t]
    if not 1 <= count <= len(pairs) or repeats < 1:
        raise ValueError(f"Pairs must be 1..{len(pairs)} and repeats must be positive.")
    chosen = random.Random(seed).sample(pairs, count)
    oracle = exact_distances(graph)
    rows = []
    for start, goal in chosen:
        for name, algorithm in ALGORITHMS.items():
            algorithm(graph, start, goal)  # one untimed warm-up per query/algorithm
            latencies = []
            for _ in range(repeats):
                begin = perf_counter_ns()
                result = algorithm(graph, start, goal)
                latencies.append((perf_counter_ns() - begin) / 1_000_000)
            validate_result(graph, start, goal, result,
                            oracle[start][goal] if name in GUARANTEED else None)
            expected = oracle[start][goal]
            reached = bool(result["path"])
            optimal = reached and math.isclose(result["cost"], expected, abs_tol=1e-7)
            gap = max(0.0, 100 * (result["cost"] / expected - 1)) if reached and expected else None
            rows.append({"algorithm": name, "start": start, "goal": goal,
                         "cost_model_m": round(result["cost"], 6) if reached else None,
                         "oracle_cost_model_m": round(expected, 6) if math.isfinite(expected) else None,
                         "expanded": result["expanded"],
                         "median_runtime_ms": round(statistics.median(latencies), 6),
                         "reached": reached, "optimal": optimal,
                         "gap_percent": round(gap, 6) if gap is not None else None})
    summary = []
    for name in ALGORITHMS:
        group = [r for r in rows if r["algorithm"] == name]
        reachable = [r for r in group if r["reached"]]
        summary.append({"algorithm": name, "queries": count, "reached": len(reachable),
                        "optimal_queries": sum(r["optimal"] for r in group),
                        "mean_cost_model_m": round(statistics.mean(r["cost_model_m"] for r in reachable), 3) if reachable else None,
                        "mean_expanded": round(statistics.mean(r["expanded"] for r in group), 3),
                        "mean_median_runtime_ms": round(statistics.mean(r["median_runtime_ms"] for r in group), 6),
                        "mean_gap_percent": round(statistics.mean(r["gap_percent"] for r in reachable), 3) if reachable else None,
                        "max_gap_percent": round(max(r["gap_percent"] for r in reachable), 3) if reachable else None})
    write_csv(output / "benchmark_queries.csv", rows)
    write_csv(output / "benchmark_summary.csv", summary)
    metadata = {"seed": seed, "pairs": count, "repeats": repeats,
                "timing": "Mean of per-query medians, one untimed warm-up, search only including trace construction",
                "python": sys.version.split()[0], "platform": platform.platform(),
                "nodes": len(graph["nodes"]), "directed_arcs": sum(map(len, graph["edges"].values())),
                "scale_model_m_per_reference_pixel": graph["scale"],
                "units": graph["units"], "summary": summary}
    print("\n50-pair comparison" if count == 50 else f"\n{count}-pair comparison")
    print(f"{'Algorithm':25} {'Optimal':>8} {'Expands':>9} {'ms':>10} {'Mean gap %':>11}")
    for row in summary:
        gap = row["mean_gap_percent"]
        print(f"{row['algorithm']:25} {row['optimal_queries']:>3}/{count:<4} "
              f"{row['mean_expanded']:9.3f} {row['mean_median_runtime_ms']:10.6f} "
              f"{gap if gap is not None else 'N/A':>11}")
    return metadata


def resolve_node(graph, text):
    text = text.strip()
    if text.upper() in graph["nodes"]:
        return text.upper()
    for node, record in graph["nodes"].items():
        if text.casefold() == record["name"].casefold():
            return node
    raise ValueError(f"Unknown landmark '{text}'. Use --list for IDs.")


def show_route(graph, start, goal, algorithm, output):
    begin = perf_counter_ns()
    result = ALGORITHMS[algorithm](graph, start, goal)
    latency = (perf_counter_ns() - begin) / 1_000_000
    print(f"\n{algorithm}: {start} -> {goal}")
    if not result["path"]:
        print("No route is reachable using the available directed paths.")
    else:
        print(" -> ".join(result["path"]))
        print(f"Distance: {result['cost']:.3f} estimated model metres")
    print(f"Node expansions: {result['expanded']}; search runtime: {latency:.6f} ms")
    if algorithm not in GUARANTEED:
        print("This algorithm has no shortest-path guarantee. Compare its cost with UCS.")
    write_csv(output / "search_trace.csv", result["trace"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="Node ID or full display name")
    parser.add_argument("--goal", help="Node ID or full display name")
    parser.add_argument("--algorithm", choices=list(ALGORITHMS), default="A* Euclidean")
    parser.add_argument("--list", action="store_true", help="List landmark and junction IDs")
    parser.add_argument("--interactive", action="store_true", help="Repeated route queries (default for a normal run)")
    parser.add_argument("--demo", action="store_true", help="Run the fixed MAIN_GATE to SJT example")
    parser.add_argument("--verify", action="store_true", help="Run independent correctness checks")
    parser.add_argument("--benchmark", action="store_true", help="Run the comparison experiment")
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=1.0, help="Estimated model metres per reference pixel (default 1, illustrative)")
    parser.add_argument("--close", action="append", default=[], help="Close both arcs, e.g. S3_S:S3_N")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    # A normal run should ask the student for inputs. Automated experiment,
    # verification, list and explicitly selected route commands stay noninteractive.
    if not (args.start or args.goal or args.list or args.verify or args.benchmark or args.demo):
        args.interactive = True
    try:
        graph = build_graph(args.scale, args.close)
        if bool(args.start) != bool(args.goal):
            raise ValueError("Provide both --start and --goal.")
        if args.pairs < 1 or args.repeats < 1:
            raise ValueError("Pairs and repeats must be positive.")
        # Resolve before writing output, so an invalid user query fails cleanly.
        start = resolve_node(graph, args.start) if args.start else "MAIN_GATE"
        goal = resolve_node(graph, args.goal) if args.goal else "SJT"
        if args.list:
            for node, record in graph["nodes"].items():
                kind = "landmark" if record["landmark"] else "junction"
                print(f"{node:15} {record['name']} ({kind})")
            if not (args.verify or args.benchmark or args.start or args.interactive or args.demo):
                return
        args.output.mkdir(parents=True, exist_ok=True)
        save_graph(graph, args.output / "campus_graph.json")
        verification = verify(graph) if args.verify else None
        if verification and not args.benchmark:
            (args.output / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
        if args.benchmark:
            metadata = benchmark(graph, args.output, args.pairs, args.repeats, args.seed)
            metadata["closed_roads"] = args.close
            metadata["verification"] = verification
            (args.output / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if args.start or args.demo:
            show_route(graph, start, goal, args.algorithm, args.output)
        if args.interactive:
            print("\nCampus route finder - enter your own start and destination.")
            print(f"Algorithm: {args.algorithm}")
            print("Available landmark IDs:")
            for node, record in graph["nodes"].items():
                if record["landmark"]:
                    print(f"  {node:15} {record['name']}")
            print("Junction IDs from --list also work. Type q to quit.")
            while True:
                try:
                    entered_start = input("Start: ").strip()
                    if entered_start.casefold() == "q":
                        break
                    selected_start = resolve_node(graph, entered_start)
                    entered_goal = input("Goal: ").strip()
                    if entered_goal.casefold() == "q":
                        break
                    selected_goal = resolve_node(graph, entered_goal)
                    show_route(graph, selected_start, selected_goal,
                               args.algorithm, args.output)
                except ValueError as error:
                    print(error)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print("\nDemo ended.")
    except EOFError:
        print("\nNo keyboard input available. Run in a terminal or use --start LIBRARY --goal PRP.")


if __name__ == "__main__":
    main()
