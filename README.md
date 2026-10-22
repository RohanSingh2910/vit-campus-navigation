# Multi-Heuristic A* Campus Navigation and Optimal Route Finder

BCSE306L Artificial Intelligence, Module 2. A simple command-line project based on the supplied VIT Vellore map and team proposal. The AI core, binary min-heap, heuristics and Greedy search are written explicitly in Python. No AI service, NetworkX, heapq or machine-learning library runs the search.

**Problem:** find the least-cost route between two road access points in a manually modelled campus pedestrian graph.

**Important data assumption:** the supplied image has no distance scale. The default `--scale 1` means one pixel in the 1882 x 1344 reference image equals one **estimated model metre**, purely for the experiment. These are not measured walking distances. Landmarks mark approximate road access points. Road links, hostel access, subway access and permissions require an on-campus check. Optimality is a guarantee about this graph, not a guarantee that every modelled path is accessible in real life.

## Six Python files in sequence

| Order | File | Assigned member and task |
|---|---|---|
| 1 | `step01_graph_Tanishq.py` | Tanishq (24BCE0980): map points, road bends, positive weights, closures, dataset generation |
| 2 | `step02_heap_Rohan.py` | Rohan (24BCE0985): custom min-heap, push/pop and sift operations |
| 3 | `step03_heuristics_Rahul.py` | Rahul (24BCE3006): zero, Euclidean, raw and scaled Manhattan, consistency checks |
| 4 | `step04_astar_Nimisha.py` | Nimisha (24BCT0188): A*, route reconstruction, frontier tracking and reopening |
| 5 | `step05_baselines_Ishaan.py` | Ishaan (24BDS0243): UCS and Greedy comparison baselines |
| 6 | `step06_run_Tanishq.py` | Tanishq: demo, benchmarking, independent verification and comparison tables |

The `stepNN_task_Person` names are valid Python module names. This lets us use ordinary imports. The table assigns review responsibilities from the proposal. It does not claim the members independently authored this generated draft. Everyone must understand the whole core before the individual viva.

## Exact run instructions

Use Python **3.9 or newer**. Run commands from the folder containing the six files. On Windows, replace `python3` with `python` if necessary.

The route finder and CSV experiment need only Python's standard library:

```bash
python3 step06_run_Tanishq.py
python3 step06_run_Tanishq.py --list
python3 step06_run_Tanishq.py --start LIBRARY --goal PRP
python3 step06_run_Tanishq.py --interactive
```

Running `python3 step06_run_Tanishq.py` now lists the available landmarks and
prompts for `Start:` and `Goal:`. Enter IDs such as `LIBRARY` and `PRP`. After
showing the result, it asks for another route. Type `q` to finish. Enter these
values in your terminal or IDE's interactive console.

The earlier version automatically ran `MAIN_GATE -> SJT`. To run that fixed
example explicitly, use `python3 step06_run_Tanishq.py --demo`. Executing
`step04_astar_Nimisha.py` or `step05_baselines_Ishaan.py` directly still runs
their small `MAIN_GATE -> SJT` checks. Use the step 6 runner for user input.

The experiment (`--benchmark`) is separate from the input demo. It selects
50 distinct ordered landmark pairs with seed 42 and runs all five methods
on those same pairs. The exact start and goal of every case appear in
`output/benchmark_queries.csv`; each pair has five rows, one per method.
The verification command (`--verify`) checks every graph-node pair and
additional constructed cases. Neither command requires you to type routes.

No package installation is needed. Python 3.9+ is sufficient. The professor allows a table or a plot for the comparison, so this version uses terminal and CSV tables only.

**One command for the full experiment and demo:**

```bash
python3 step06_run_Tanishq.py --verify --benchmark --start MAIN_GATE --goal SJT
```

To explain each member's code, execute the files in order:

```bash
python3 step01_graph_Tanishq.py
python3 step02_heap_Rohan.py
python3 step03_heuristics_Rahul.py
python3 step04_astar_Nimisha.py
python3 step05_baselines_Ishaan.py
python3 step06_run_Tanishq.py --verify --benchmark --start MAIN_GATE --goal SJT
```

The final file imports the earlier ones automatically. They do not need to remain running, and the route finder regenerates the graph in memory. `campus_graph.json` is an inspectable exported dataset, not a separate input that you must edit.

## Professor-selected inputs and closures

Any node listed by `--list` can be the start or goal, including junctions. IDs are case-insensitive. A full display name also works in quotes.

```bash
python3 step06_run_Tanishq.py --start GATE11 --goal MGR
python3 step06_run_Tanishq.py --start "Library access" --goal "SJT access"
python3 step06_run_Tanishq.py --start SJT --goal SJT
python3 step06_run_Tanishq.py --start MAIN_GATE --goal MH_N --close S3_S:S3_N --output output/closure_demo
python3 step06_run_Tanishq.py --start MAIN_GATE --goal SJT --algorithm "Greedy Euclidean" --output output/greedy_demo
python3 step06_run_Tanishq.py --start MAIN_GATE --goal SJT --algorithm "A* Raw Manhattan" --output output/raw_demo
```

`--close FROM:TO` removes both directed arcs of one modelled road. Repeat the flag for additional closures. For a clean unreachable example, close the Main Gate's only connection:

```bash
python3 step06_run_Tanishq.py --start MAIN_GATE --goal SJT --close MAIN_GATE:MGR --output output/unreachable_demo
```

Unknown nodes, malformed closures and invalid scales produce clear errors. `--help` lists options. An alternate output folder keeps closure and demonstration results separate from the submitted benchmark snapshot.

## AI formulation

Let `G=(V,E)` be a directed, weighted graph with positive edge costs. The default dataset has 52 vertices and 126 arcs (63 roads, each stored in both directions), including 30 selectable landmarks and 22 junctions.

- State: a vertex `v in V` representing a road access point or junction.
- Initial state: the selected start vertex `s`.
- Actions: traverse one available outgoing pedestrian arc `(v,u)`.
- Transition: `Result(v, traverse(v,u)) = u`, deterministically.
- Goal test: `v == t`, where `t` is the selected destination.
- Cost: `g(u) = g(v) + c(v,u)`. A route's cost is the sum of its edge costs.
- Objective: minimize total modelled distance. Expansions and runtime measure the effort used to find that route, not additional terms in the objective.

Road cost is the length of its manually traced polyline multiplied by scale, rounded **up** to 0.001 model metre. The upward rounding preserves `c(u,v) >= Euclidean(u,v)`. The graph generator validates this condition. The railway has only four explicit subway connections, and there is no edge across the lake.

## Where the AI makes a decision

In `step04_astar_Nimisha.py`, `frontier.pop()` chooses the smallest `f=g+h` entry. The `candidate < best_g.get(neighbor, infinity)` decision accepts a cheaper known route, updates its parent and inserts its new priority.

```text
put start into the min-heap with priority h(start)
best_g[start] = 0
while heap is not empty:
    remove the entry with the smallest f = g + h
    skip stale entries or entries already expanded at a better cost
    if this node is the goal: reconstruct and return its path
    record an expansion
    for each outgoing neighbor:
        candidate = current_g + edge_cost
        if candidate improves best_g[neighbor]:
            update best_g and parent
            insert (candidate + h(neighbor), sequence, neighbor, candidate)
return unreachable
```

The heap uses insertion sequence as FIFO tie-breaking. A node may have old heap entries after a better path is found. Lazy decrease-key skips those entries when popped. Reopening permits a better path to revisit an expanded node. The goal test occurs after a valid heap pop, not when the goal is first discovered. An expansion counts generation of successors; the final goal pop is excluded. `search_trace.csv` includes the valid goal pop and shows `g`, `h` and `f`.

## Heuristics and shortest-path guarantees

| Method | Heap priority | Shortest-path guarantee in this model |
|---|---|---|
| UCS / A* with zero heuristic | `g` | Yes |
| Greedy Euclidean | `h_E` | No |
| A* Euclidean | `g + h_E` | Yes |
| A* scaled Manhattan | `g + h_M / sqrt(2)` | Yes |
| A* raw Manhattan | `g + h_M` | No, experimental comparison |

`h_E = scale * sqrt(dx^2 + dy^2)` and `h_M = scale * (abs(dx)+abs(dy))`. All heuristics use the same scale as edge costs.

Euclidean consistency follows from triangle inequality: `h_E(u) <= distance(u,v) + h_E(v) <= c(u,v) + h_E(v)`. Since `h(goal)=0`, consistency implies admissibility along any route to the goal.

Scaled Manhattan is also a metric, with `h_M(u,v)/sqrt(2) <= Euclidean(u,v)`. Applying its triangle inequality and the edge lower bound proves its consistency. It is never larger than Euclidean, so it is a weaker lower bound on this graph. Raw Manhattan can exceed a diagonal walking distance and is not guaranteed admissible.

A minimal counterexample used in verification is `S=(0,0), A=(1,1), G=(2,2)`. Edges `S->A=1.415`, `A->G=1.415`, `S->G=3`. The shortest route costs 2.83, but raw Manhattan gives A priority `1.415+2=3.415`. It pops G with cost 3 first. Reopening cannot correct an overestimate after early goal termination.

UCS deliberately calls the A* engine with `h=0`; this is its mathematical definition and avoids duplicate code. Floyd-Warshall is a separate exact verification oracle, not the navigation algorithm. The shortest-path guarantees assume this finite positive-cost graph and the stated heuristic conditions.

## Experiment and results

`--benchmark` samples **50 distinct ordered landmark pairs without replacement**, seed 42, excluding identical start/goal pairs. Every algorithm receives the same pairs. Each query has one untimed warm-up and 21 timed searches. Reported latency is the mean of the 50 per-query median runtimes. Timing covers search, route reconstruction and trace creation; file writing is outside the timer. Run on Python 3.14.7, Linux. Runtime will vary on your laptop.

| Algorithm | Optimal pairs / 50 | Mean expansions | Mean gap above optimum |
|---|---:|---:|---:|
| UCS | 50 | 30.720 | 0.000% |
| Greedy Euclidean | 29 | 8.640 | 8.259% |
| A* Euclidean | 50 | 13.780 | 0.000% |
| A* scaled Manhattan | 50 | 16.480 | 0.000% |
| A* raw Manhattan | 42 | 10.520 | 0.160% |

Euclidean A* reduced mean expansions by **55.1%** relative to UCS while preserving the shortest cost. Greedy used fewer expansions but missed the optimum on 21 pairs. Its worst observed cost gap was 70.018%. Raw Manhattan missed the optimum on 8 pairs, with a worst gap of 4.277%. Fewer expansions alone do not establish a better route finder. On this small dataset, timing differences are tiny; node expansions are the more stable evidence. The generated CSVs and `experiment.json` are the authoritative run data if you rerun with changed settings.

Run a different reproducible experiment with:

```bash
python3 step06_run_Tanishq.py --benchmark --pairs 50 --repeats 21 --seed 7 --output output/seed7
```

Verification checks all 2,704 ordered graph-node pairs, including equal-node cases, for each of the five methods (13,520 searches). It independently verifies UCS and the two safe A* costs against Floyd-Warshall, and verifies returned path legality and cost for every method. It also checks 980 searches on random directed graphs, heap operations, reopening, unreachable nodes, invalid IDs, closures and the diagonal counterexample. `--verify` uses Python assertions: do not run it with `python -O`.

The full campus audit found 302 suboptimal ordered pairs with raw Manhattan and 852 with Greedy. Euclidean, zero and scaled Manhattan had no consistency or admissibility violations. Raw Manhattan had 1,141 consistency violations across arc/goal triples and 1,236 admissibility violations across node/goal pairs. These exhaustive graph figures are separate from the 50-pair benchmark.

## Output files

All generated data and evidence live in `output/`:

- `campus_graph.json`: complete reproducible dataset, coordinates, arcs, road shapes and units.
- `benchmark_queries.csv`: 250 rows containing costs, oracle costs, expansions, latency and optimality.
- `benchmark_summary.csv` and `experiment.json`: aggregates, settings, machine information and verification summary.
- `search_trace.csv`: most recent route's valid pop sequence, including the goal.
- `Project_Report.pdf`: 10-page report covering all requested sections.
- `Presentation.pdf`: exactly six slides for the final review; `Presentation.pptx` is the editable version with speaker notes.

The last two are prepared submission documents using the recorded experiment. Rerunning the benchmark regenerates data tables but does not rewrite these documents. Update their tables if you change the model or experiment. PDF slides are the required submission format; the editable presentation is also supplied.

## Eight-minute demonstration

1. **0:00-1:00:** state the problem, show the map, explain the graph and estimated scale.
2. **1:00-2:00:** explain states, actions, goal test and summed path cost.
3. **2:00-3:30:** run a professor-chosen pair using `--list`, then point to `frontier.pop()` and the cheaper-route condition.
4. **3:30-5:00:** explain the heap, `g+h`, lazy entries and Euclidean's lower bound.
5. **5:00-6:30:** show the CSV comparison and explain why raw Manhattan and Greedy lose optimality.
6. **6:30-7:30:** show a closure, a same-node query and an unreachable case.
7. **7:30-8:00:** summarize limitations and state the AI assistance declaration.

## Individual viva preparation

Every member should answer these in their own words:

| Question | Main point |
|---|---|
| What is `g`, `h`, `f`? | Actual discovered cost, estimated remaining cost, and `g+h`. |
| Why test the goal when popped? | Finding a goal neighbor does not yet prove its route is cheapest. |
| Why is UCS a baseline? | Setting `h=0` searches by accumulated cost and gives the shortest cost. |
| Why is Greedy sometimes wrong? | It ignores distance already travelled, so closeness to the goal can be misleading. |
| What is admissibility? | `h(n)` never exceeds the true cheapest remaining cost. |
| What is consistency? | For every arc, `h(u) <= c(u,v)+h(v)`. |
| Why does the heap help? | Insert and remove-min take logarithmic time instead of scanning the whole frontier. |
| Why not use BFS? | BFS minimizes edge count, not total distance on unequal weights. |
| Why can a node appear twice in the heap? | A cheaper route creates a new entry; the older entry becomes stale. |
| Why reopen? | An admissible but inconsistent heuristic can reveal a cheaper route after expansion. |
| Does map overlay prove road validity? | No. It shows the model; real paths and permissions need field validation. |
| How are equal priorities handled? | Earlier insertion wins, deterministically. |
| What is the AI decision? | Choosing the minimum frontier priority and accepting a lower-cost candidate route. |

For a consistent heuristic, search with this lazy heap takes roughly `O((V+E) log E)` worst-case time and `O(V+E)` auxiliary memory, since stale entries can remain. An inconsistent heuristic with reopening can require additional expansions. Floyd-Warshall verification uses `O(V^3)` time and `O(V^2)` memory, appropriate only for this small graph.

## Limitations and next work

This is a manually digitised teaching model. It does not implement live GPS, automatic image segmentation, measured metres, stairs, wheelchair access, crowding or individual access permissions. Collision avoidance means only traversing listed graph arcs; there is no geometric obstacle-detection algorithm. Trace bends and a single uniform scale cannot capture satellite distortion. Walk the selected paths, measure several known distances, then correct `POINTS`, `ROADS` and the scale. If measured edge weights fall below straight-line estimates, recalibrate coordinates/scale before using the safe heuristics. Run verification again after any change.

## References and external AI declaration

- Supplied course proposal: team title, roles and comparison plan.
- Supplied VIT Vellore map, Office of Students' Welfare: manual visual digitisation. Date, scale and survey accuracy are unspecified.
- Professor Nancy Lydia R's supplied project guidelines: required algorithm, report, comparison and individual viva.
- UC Berkeley CS188, [Informed Search](https://inst.eecs.berkeley.edu/~cs188/textbook/search/informed.html): heuristic reasoning and A* fundamentals.
- Russell and Norvig, [AIMA informed-search teaching slides](https://aima.cs.berkeley.edu/slides-pdf/chapter04a.pdf): UCS, Greedy and A* priorities.

**Declaration:** OpenAI Codex was used to draft substantial portions of the Python implementation, approximate map digitisation, verification cases, report and presentation. No external AI model or API performs routing during execution. The code was executed and checked against an independent shortest-path oracle. Team members must inspect, understand, correct and validate this draft before submission, and retain this disclosure rather than claiming unsupported independent authorship or surveyed data.
