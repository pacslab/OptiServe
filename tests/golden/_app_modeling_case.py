"""Golden-master case: the App graph from experiments/application_modeling.ipynb.
Deterministic analytical outputs (ne, end-to-end RT) — no AWS required.
Import-path-agnostic so it runs before AND after the src->optiserve rename.
"""

import networkx as nx


def build_graph():
    G = nx.DiGraph()
    mem = {1: 512, 2: 1024, 3: 2048, 4: 4096, 5: 128, 6: 256}
    rt = {1: 100, 2: 500, 3: 1000, 4: 210, 5: 130, 6: 100}
    G.add_node("Start", pos=(0, 0))
    for n in range(1, 7):
        G.add_node(n, mem=mem[n], rt=rt[n])
    G.add_node("End", pos=(5, 0))
    G.add_weighted_edges_from(
        [
            ("Start", 1, 1.0),
            (1, 2, 0.2),
            (1, 3, 0.8),
            (1, 4, 1.0),
            (2, 5, 1.0),
            (3, 5, 1.0),
            (4, 5, 1.0),
            (5, 6, 0.7),
            (6, 6, 0.2),
            (6, "End", 0.8),
            (5, 1, 0.3),
        ]
    )
    return G


def compute(ApplicationPerformanceModeling):
    app = ApplicationPerformanceModeling(build_graph(), delay_type="None")
    app.update_ne()
    ne = {str(k): round(float(v), 6) for k, v in app.ne.items()}
    app.get_simple_dag()
    rt = round(float(app.get_avg_rt()), 6)
    return {"ne": ne, "rt": rt}
