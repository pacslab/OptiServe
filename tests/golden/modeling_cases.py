"""A battery of application-model graph shapes for golden-master regression.

Each builder returns a networkx DiGraph with Start/End sentinels, node `mem`/`rt`
attributes, and edge `weight` (transition probability). `compute_all` evaluates
expected executions (ne) and end-to-end response time for every case, under both
delay models, so a refactor of the graph-reduction code can be proven behavior-
preserving.
"""
from __future__ import annotations

import networkx as nx


def _g(nodes, edges):
    G = nx.DiGraph()
    G.add_node("Start")
    for n, (mem, rt) in nodes.items():
        G.add_node(n, mem=mem, rt=rt)
    G.add_node("End")
    G.add_weighted_edges_from(edges)
    return G


def case_mixed():
    """Branch + parallel fan-out + self-loop + backward edge (the notebook app)."""
    nodes = {1: (512, 100), 2: (1024, 500), 3: (2048, 1000), 4: (4096, 210),
             5: (128, 130), 6: (256, 100)}
    edges = [("Start", 1, 1.0), (1, 2, 0.2), (1, 3, 0.8), (1, 4, 1.0),
             (2, 5, 1.0), (3, 5, 1.0), (4, 5, 1.0), (5, 6, 0.7),
             (6, 6, 0.2), (6, "End", 0.8), (5, 1, 0.3)]
    return _g(nodes, edges)


def case_parallel():
    """Two deterministic (tp=1) parallel paths that join — max-latency semantics."""
    nodes = {1: (512, 50), 2: (512, 300), 3: (512, 700), 4: (512, 40)}
    edges = [("Start", 1, 1.0), (1, 2, 1.0), (1, 3, 1.0),
             (2, 4, 1.0), (3, 4, 1.0), (4, "End", 1.0)]
    return _g(nodes, edges)


def case_branch():
    """A probabilistic branch whose weights sum to 1 — expected-value semantics."""
    nodes = {1: (512, 100), 2: (512, 400), 3: (512, 900)}
    edges = [("Start", 1, 1.0), (1, 2, 0.3), (1, 3, 0.7),
             (2, "End", 1.0), (3, "End", 1.0)]
    return _g(nodes, edges)


def case_self_loop():
    """A single self-looping node."""
    nodes = {1: (512, 100), 2: (512, 200)}
    edges = [("Start", 1, 1.0), (1, 2, 1.0), (2, 2, 0.25), (2, "End", 0.75)]
    return _g(nodes, edges)


CASES = {
    "mixed": case_mixed,
    "parallel": case_parallel,
    "branch": case_branch,
    "self_loop": case_self_loop,
}


def compute_all(ApplicationPerformanceModeling):
    out = {}
    for name, builder in CASES.items():
        for delay in ("None", "SFN"):
            app = ApplicationPerformanceModeling(builder(), delay_type=delay)
            app.update_ne()
            ne = {str(k): round(float(v), 6) for k, v in app.ne.items()}
            app.get_simple_dag()
            rt = round(float(app.get_avg_rt()), 6)
            out[f"{name}:{delay}"] = {"ne": ne, "rt": rt}
    return out
