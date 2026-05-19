import itertools

import networkx as nx
import numpy as np

from optiserve.config import ModelingConfig
from optiserve.cost import CostCalculator
from optiserve.logging import get_logger
from optiserve.modeling.cache import make_cache
from optiserve.modeling.graph_reduction import CachedGraphAnalysis

logger = get_logger(__name__)


class ApplicationPerformanceModeling:
    """Analytical model of a serverless workflow's expected end-to-end response
    time, cost, and per-function execution counts.

    ``delay_type`` selects the per-node/per-edge transition overhead model:
    ``'None'`` (0), ``'SFN'`` (AWS Step Functions overheads from
    :class:`~optiserve.config.ModelingConfig`), or ``'Defined'`` (read from
    ``'delay'`` node/edge attributes).
    """

    def __init__(self, graph, delay_type='SFN', config: ModelingConfig = None,
                 cost_calculator: CostCalculator = None, cache_evaluations: bool = False):
        self._config = config or ModelingConfig()

        # Never mutate the caller's graph. The sentinel attributes below used to
        # be written onto the argument itself, so constructing a model silently
        # rewrote the caller's Start/End nodes — and a graph reused for a second
        # model, or inspected afterwards, carried those edits. Copy first.
        graph = graph.copy()

        if ('Start' in graph.nodes and 'End' in graph.nodes):
            graph.nodes['Start']['rt'] = 0
            graph.nodes['End']['rt'] = 0
            graph.nodes['Start']['mem'] = 0
            graph.nodes['End']['mem'] = 0
            graph.nodes['Start']['perf_profile'] = {0: 0}
            graph.nodes['End']['perf_profile'] = {0: 0}

        else:
            raise ValueError('Graph must contain both "Start" and "End" nodes.')

        self.workflow_graph = graph.copy()
        self.simple_dag = graph.copy()
        self.delooped_graph = graph.copy()
        self.start_point = 'Start'
        self.end_point = 'End'
        self.rt = nx.get_node_attributes(self.workflow_graph, 'rt')
        self.dag_rt = nx.get_node_attributes(self.simple_dag, 'rt')

        if (delay_type == 'None'):
            self.node_delay = {node: 0 for node in self.workflow_graph.nodes}
            self.edge_delay = {edge: 0 for edge in self.workflow_graph.edges}

        elif (delay_type == 'SFN'):
            self.node_delay = {
                node: self._config.sfn_node_delay_ms
                for node in self.workflow_graph.nodes
            }
            self.edge_delay = {
                edge: self._config.sfn_edge_delay_ms
                for edge in self.workflow_graph.edges
            }

        elif (delay_type == 'Defined'):
            self.node_delay = nx.get_node_attributes(self.workflow_graph, 'delay')
            self.edge_delay = nx.get_edge_attributes(self.workflow_graph, 'delay')

        else:
            raise ValueError(
                f"Unknown delay_type {delay_type!r}; expected 'None', 'SFN', or 'Defined'."
            )

        self.p_node_num = 1
        self.b_node_num = 1
        self.mem = nx.get_node_attributes(self.workflow_graph,
                                          'mem')  # Memory configuration of each function in the workflow

        # Injectable so the analytical layer can run with fixed unit prices and
        # never touch the AWS Price List API (offline evaluation, tests, CI).
        self.cost_calculator = cost_calculator or CostCalculator()

        self.ne = {}
        self.approximations = []

        # -- evaluation memoization (opt-in) -------------------------------- #
        # Keys are the exact float vectors, so a hit is bit-identical to the
        # recomputation. Off by default: enabling it also skips the repeated
        # `approximations` bookkeeping and synthetic P#/B# numbering that a
        # recomputation would redo, which is diagnostic output some callers read.
        self._node_order = tuple(self.workflow_graph.nodes)
        self._ne_version = 0
        self._rt_cache = make_cache(cache_evaluations)
        self._cost_cache = make_cache(cache_evaluations)
        # Which configuration `self.simple_dag` currently describes. A cached
        # evaluate_avg_rt() skips the reduction, so the reduced graph is left
        # describing whatever was evaluated last; this lets get_avg_rt() detect
        # that and rebuild instead of silently answering from a stale graph.
        self._reduction_key = None
        # Cycle discovery and the `is_simple` predicate depend only on topology
        # and edge weights, never on rt/mem — so they repeat identically across
        # every configuration the optimizer scores. Together they are ~54% of a
        # ground-truth sweep's runtime (measured); this is where that goes.
        self._graphs = CachedGraphAnalysis(make_cache(cache_evaluations))


    def update_rt(self):
        self.rt = nx.get_node_attributes(self.workflow_graph, 'rt')
        self.dag_rt = nx.get_node_attributes(self.simple_dag, 'rt')


    def remove_paths(self, paths):
        if (type(paths[0]) == list):
            for path in paths:
                for i in range(1, len(path) - 1):
                    self.simple_dag.remove_edge(path[i], path[i + 1])
                    self.simple_dag.remove_node(path[i])
        else:
            for i in range(1, len(paths) - 1):
                self.simple_dag.remove_edge(paths[i], paths[i + 1])
                self.simple_dag.remove_node(paths[i])


    # Calculate the sum of RT of each path in a set of paths. (start node and end node not included by default)
    def sum_rt(self, paths, include_start_node=False, include_end_node=False, include_first_edge_delay=False,
              include_last_edge_delay=False):
        start = 1 - include_start_node
        end = -1 if not include_end_node else None
        start_edge = 1 - include_first_edge_delay
        end_edge = -1 if not include_last_edge_delay else None

        if (type(paths[0]) == list):
            rt_results = []

            for path in paths:
                edges = [(first, second) for first, second in
                         zip(path[start_edge: end_edge], path[start_edge + 1:end_edge])]
                edge_delay = sum([self.edge_delay[edge] for edge in edges])
                rt_results.append(
                    sum([self.dag_rt.get(name) + self.node_delay[name] for name in path[start:end]]) + edge_delay)

            return rt_results

        else:
            edges = [(first, second) for first, second in
                     zip(paths[start_edge: end_edge], paths[start_edge + 1:end_edge])]
            edge_delay = sum([self.edge_delay[edge] for edge in edges])

            return sum([self.dag_rt.get(name) + self.node_delay[name] for name in paths[start:end]]) + edge_delay


    # Calculate the sum of RT of each path in a given graph.
    def sum_rt_with_ne(self, paths, include_start_node=False, include_end_node=False):
        self.rt = nx.get_node_attributes(self.workflow_graph, 'rt')
        self.ne = nx.get_node_attributes(self.workflow_graph, 'ne')
        start = 1 - include_start_node
        end = -1 if not include_end_node else None

        if (type(paths[0]) == list):
            return list(map(lambda x: sum([self.rt.get(name) * self.ne.get(name) for name in x[start:end]]), paths))

        else:
            return sum([self.rt.get(name) * self.ne.get(name) for name in paths[start:end]])


    # Calculate the product of TP of each path in a set of paths.
    def get_tp(self, graph, paths):
        if (paths == []):
            return []

        if (type(paths[0]) == list):
            res = []

            for path in map(nx.utils.pairwise, paths):
                res.append(np.prod(list(map(lambda edge: graph.get_edge_data(edge[0], edge[1])['weight'], list(path)))))

            return res

        else:
            return self.get_tp(graph, [paths])[0]


    # Get the number of executions of each function in the workflow.
    def get_avg_ne(self, graph, start_point):
        nx.set_node_attributes(graph, 1, name='ne')
        ne = nx.get_node_attributes(graph, 'ne')

        while [item for item in nx.simple_cycles(graph)] != []:

            self_loops = [item for item in nx.selfloop_edges(graph, data=True)]
            self_loops = [(item[0], item[2]['weight'], ne[item[0]]) for item in self_loops]

            for self_loop in self_loops:
                out_edges = {item: graph.get_edge_data(item[0], item[1])['weight'] for item in
                             graph.out_edges(self_loop[0])}
                tp_list = [out_edges[item] for item in out_edges]

                if (round(np.sum(tp_list), 8) != 1.0):
                    raise Exception(f'Invalid Self-loop: Node:{self_loop[0]}')

                new_ne = self_loop[2] / (1 - self_loop[1])
                graph.nodes[self_loop[0]]['ne'] = new_ne
                graph.remove_edge(self_loop[0], self_loop[0])
                del out_edges[(self_loop[0], self_loop[0])]
                tp_denominator = 1.0 - self_loop[1]

                for edge in out_edges:
                    graph[edge[0]][edge[1]]['weight'] = out_edges[edge] / tp_denominator

            ne = nx.get_node_attributes(graph, 'ne')
            cycles = self._graphs.discover_cycles(graph, self.start_point)
            cycles_dict = {}

            for item in cycles:

                if (graph.has_edge(item[-1], item[0]) and (item[0], item[-1]) not in cycles_dict.keys()):
                    cycles_dict[(item[0], item[-1])] = graph.get_edge_data(item[-1], item[0])['weight']

            for key in cycles_dict:

                nodes_in_cycle = list(set([item for item in itertools.chain.from_iterable(
                    [item for item in nx.all_simple_paths(graph, key[0], key[1])])]))
                out_edges = {item: graph.get_edge_data(item[0], item[1])['weight'] for item in graph.out_edges(key[1])}
                tp_list = [out_edges[item] for item in out_edges]

                if (round(np.sum(tp_list), 4) > 1.0):
                    raise Exception(f'Invalid Loop: Node:{key[1]}')

                for node in nodes_in_cycle:
                    graph.nodes[node]['ne'] = graph.nodes[node]['ne'] / (1.0 - cycles_dict[key])

                graph.remove_edge(key[1], key[0])
                del out_edges[(key[1], key[0])]
                tp_denominator = 1.0 - cycles_dict[key]

                for edge in out_edges:
                    graph[edge[0]][edge[1]]['weight'] = out_edges[edge] / tp_denominator

                ne = nx.get_node_attributes(graph, 'ne')

        for node in graph.nodes:

            paths = [item for item in nx.all_simple_paths(graph, start_point, node)]
            sum_tp = round(np.sum(self.get_tp(graph, paths)), 4)

            if (node == start_point):
                sum_tp = 1

            if sum_tp < 1.0:
                graph.nodes[node]['ne'] = graph.nodes[node]['ne'] * sum_tp

        return (graph, nx.get_node_attributes(graph, 'ne'))


    def update_ne(self):
        graph, ne = self.get_avg_ne(self.workflow_graph.copy(), self.start_point)
        ne = {key: {'ne': ne[key]} for key in ne}
        ne['Start'] = {'ne': 0}
        ne['End'] = {'ne': 0}
        nx.set_node_attributes(self.workflow_graph, ne)
        self.ne = nx.get_node_attributes(self.workflow_graph, 'ne')
        self.delooped_graph = graph
        # Execution counts feed the cost model, so any cached cost keyed on the
        # previous `ne` vector is now stale.
        self._ne_version += 1
        self._cost_cache.clear()


    # ------------------------------------------------------------------ #
    # Memoized evaluation
    #
    # `get_simple_dag()` + `get_avg_rt()` is the model's hot path: the greedy
    # optimizer and the brute-force sweep call it once per candidate
    # configuration, and it re-derives the whole reduction every time. Both
    # results are pure functions of the per-node rt/mem vectors, so an exact-key
    # cache returns the identical float without recomputing.
    # ------------------------------------------------------------------ #
    def _rt_key(self):
        rt = nx.get_node_attributes(self.workflow_graph, 'rt')
        return tuple(rt.get(node) for node in self._node_order)

    def _cost_key(self):
        rt = nx.get_node_attributes(self.workflow_graph, 'rt')
        mem = nx.get_node_attributes(self.workflow_graph, 'mem')
        return (
            self._ne_version,
            tuple(rt.get(node) for node in self._node_order),
            tuple(mem.get(node) for node in self._node_order),
        )

    def evaluate_avg_rt(self):
        """Reduce the graph and return the expected end-to-end response time.

        Equivalent to ``get_simple_dag(); get_avg_rt()`` — the pair every caller
        already uses — but memoized on the exact rt vector when the model was
        constructed with ``cache_evaluations=True``.
        """
        key = self._rt_key()
        found, cached = self._rt_cache.lookup(key)
        if found:
            return cached
        self.get_simple_dag()
        value = self.get_avg_rt()
        self._rt_cache.put(key, value)
        return value

    def evaluate_avg_cost(self):
        """Expected cost for the current configuration, memoized on (ne, rt, mem)."""
        key = self._cost_key()
        found, cached = self._cost_cache.lookup(key)
        if found:
            return cached
        value = self.get_avg_cost()
        self._cost_cache.put(key, value)
        return value

    def cache_stats(self):
        """Hit/miss counters for both evaluation caches."""
        return {'rt': self._rt_cache.stats.as_dict(), 'cost': self._cost_cache.stats.as_dict()}


    def get_avg_cost(self):
        num_exe = [item for item in self.ne.values()]
        self.mem = nx.get_node_attributes(self.workflow_graph, 'mem')
        self.rt = nx.get_node_attributes(self.workflow_graph, 'rt')

        mem_cost = [
            self.cost_calculator.calculate_cost(
                memory_mb=mem,
                duration_ms=rt,
                calculate_invocation_cost=True
            )
            for mem, rt in zip(self.mem.values(), self.rt.values())
        ]

        avg_cost = np.multiply(mem_cost, num_exe)

        return np.sum(avg_cost) * 1_000_000


    def get_rttp_for_paths_with_b_node(self, paths_with_b_node):
        rttp = []

        for path in paths_with_b_node:

            b_nodes = [node for node in path if type(node) == str and node[0] == 'B']
            b_node_rt_dict = {bnode: self.dag_rt.get(bnode) for bnode in b_nodes}

            for bnode in b_nodes:
                self.dag_rt[bnode] = 0

            path_b_node_zeroing_rt = self.sum_rt(paths=path, include_start_node=False, include_end_node=False,
                                                include_first_edge_delay=True, include_last_edge_delay=True)

            for bnode in b_node_rt_dict.keys():
                self.dag_rt[bnode] = b_node_rt_dict[bnode]

            if (len(b_nodes) > 1):
                rt_list = [self.simple_dag.nodes[bnode]['rt_list'] for bnode in b_nodes]
                tp_list = [self.simple_dag.nodes[bnode]['tp_list'] for bnode in b_nodes]
                rt_list = [np.sum(item) for item in itertools.product(*rt_list)]
                tp_list = [np.prod(item) for item in list(itertools.product(*tp_list))]

            else:
                rt_list = self.simple_dag.nodes[b_nodes[0]]['rt_list']
                tp_list = self.simple_dag.nodes[b_nodes[0]]['tp_list']

            rt_list = [rt + path_b_node_zeroing_rt for rt in rt_list]
            rttp.append(list(zip(rt_list, tp_list)))

        return rttp


    def process_rttp(self, rttp, rt_tp1_path):
        if (rt_tp1_path != None):
            rttp.append([(rt_tp1_path, 1)])

        tppt_product = [item for item in itertools.product(*rttp)]
        refined_rt = [(np.max([tup[0] for tup in item])) for item in tppt_product]
        refined_tp = [(round(np.prod([tup[1] for tup in item]), 4)) for item in tppt_product]

        return list(zip(refined_rt, refined_tp))


    def drawGraph(self, graph):
        import matplotlib.pyplot as plt  # lazy: plotting is an optional extra

        pos = nx.planar_layout(graph)
        nx.draw(graph, pos, with_labels=True)
        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
        pos_higher_offset = {}

        for k, v in pos.items():
            pos_higher_offset[k] = (v[0], v[1] + 0.05)

        labels = nx.get_node_attributes(graph, 'rt')
        nx.draw_networkx_labels(graph, pos_higher_offset, labels=labels, label_pos=0)
        plt.show()


    def get_parallel_paths_p1(self, graph):
        p1_path_list = []

        for u in graph.nodes():
            for v in graph.nodes():

                if u == v:
                    continue

                filtered_path = self.path_filter(graph, u, v)
                paths_tp1_filter = filter(lambda p: graph.get_edge_data(p[0], p[1])['weight'] == 1, filtered_path)
                paths_tp1 = []

                try:
                    for item in paths_tp1_filter:
                        paths_tp1.append(item)

                except KeyError:
                    # get_edge_data(...)['weight'] on an edge the filter walked
                    # past. A bare `except:` here also swallowed
                    # KeyboardInterrupt and every genuine bug in the loop body.
                    continue

                if len(paths_tp1) > 1:
                    p1_path_list += [paths_tp1]

        return p1_path_list


    def get_parallel_paths(self, graph):
        path_list = []
        for u in graph.nodes():
            for v in graph.nodes():

                if u == v:
                    continue

                filtered_path = self.path_filter(graph, u, v)

                if len(filtered_path) > 1:
                    path_list += [filtered_path]

        return path_list


    def path_filter(self, graph, u, v):
        paths = []
        for path in nx.all_simple_paths(graph, u, v):
            if len(path) == 2 or max(graph.degree(node) for node in path[1:-1]) == 2:
                paths += [path]
        return paths


    def is_simple(self):
        # Pure function of self.simple_dag's topology + edge weights, so it is
        # memoized on exactly that. The original implementation is preserved
        # verbatim inside _is_simple_uncached and remains the only copy of the
        # algorithm; the cache never becomes a second, drifting version.
        return self._graphs.memoized(
            'is_simple', self.simple_dag, self._is_simple_uncached,
            self.start_point, self.end_point,
        )


    def _is_simple_uncached(self):
        try:
            nx.find_cycle(self.simple_dag)

            return False

        except nx.NetworkXNoCycle:
            pass

        # A standalone B (branch) node is accepted as simple: its rt already
        # encodes the branch's expected value. B nodes that sit inside a
        # parallel join are resolved by simplify_parallels() (which runs in
        # get_simple_dag()'s loop before is_simple() is re-checked), so no
        # explicit B-node gate is needed here. Enabling a real B-node check
        # would loop forever on standalone B nodes. (The original code compared
        # `type(item) == 'str'`, which was always False — a no-op left in place.)

        tp_sum = 0
        paths = nx.all_simple_paths(self.simple_dag, self.start_point, self.end_point)

        for path in map(nx.utils.pairwise, paths):
            tp_sum += (
                np.prod(list(map(lambda edge: self.simple_dag.get_edge_data(edge[0], edge[1])['weight'], list(path))))
            )

        if (round(tp_sum, 4) == 1):
            return True

        else:
            return False


    def get_simple_dag(self):
        self.simple_dag = self.workflow_graph.copy()
        self.update_rt()
        # Synthetic-node counters and the approximation log are per-reduction
        # bookkeeping. Without resetting them, P#/B# names climb without bound
        # across the optimizer's hot loop and `approximations` accumulates one
        # duplicate entry per call, so `get_approximations()` reports a count
        # that grows with the number of evaluations rather than describing the
        # reduction it is being asked about.
        self.p_node_num = 1
        self.b_node_num = 1
        self.approximations = []
        self._reduction_key = self._rt_key()

        while (not self.is_simple()):

            if self.simplify_loops():
                continue
            if self.simplify_parallels():
                continue
            if self.simplify_branches():
                continue

            # No reduction pass made progress but the graph is not simple.
            # Break instead of spinning forever (safety guard for graph shapes
            # the reductions cannot handle).
            logger.warning("get_simple_dag could not further reduce the graph.")
            break


    def get_approximations(self):
        """Report the jump edges the loop reduction had to drop.

        These are the places where the analytical model is *approximate* — an
        edge escaping a cycle was removed, so the reported response time is a
        documented over- or under-estimate. That makes this the single most
        important diagnostic the model produces, and it used to be emitted with
        `print()`, so it was invisible to anything that configures logging and
        unusable from a notebook that redirects stdout.
        """
        # `self.approximations` only describes the most recent reduction, and a
        # memoized evaluate_avg_rt() may have skipped it; rebuild if the reduced
        # graph does not match the current configuration.
        if self._reduction_key != self._rt_key():
            self.get_simple_dag()

        if len(self.approximations) == 0:
            return False

        logger.warning(
            'Model made %d approximations for %d cycles.',
            len(self.approximations),
            len(set(apx['cycle'] for apx in self.approximations)),
        )

        for apx in self.approximations:
            logger.warning(
                'Jumping edge from %s to %s in the cycle between %s and %s was '
                'removed, resulting in %s RT/Cost.',
                apx['jump_from'], apx['jump_to'], apx['cycle'][0], apx['cycle'][1],
                apx['type'],
            )

        return True


    def get_avg_rt(self):
        # `self.simple_dag` is only meaningful for the configuration it was
        # reduced from. A cache hit in evaluate_avg_rt() deliberately skips the
        # reduction, so guard against answering from a graph that describes a
        # different configuration.
        if self._reduction_key != self._rt_key():
            self.get_simple_dag()
        paths = nx.all_simple_paths(self.simple_dag, self.start_point, self.end_point)

        return np.sum([self.sum_rt(item, include_start_node=True, include_end_node=True, include_first_edge_delay=True,
                                  include_last_edge_delay=True) * self.get_tp(self.simple_dag, item) for item in paths])


    def merge_parallel_paths(self, parallel_paths):
        paths_tp1_sum_rt = list(map(
            lambda x: self.sum_rt(paths=x, include_start_node=False, include_end_node=False, include_first_edge_delay=True,
                                 include_last_edge_delay=True), parallel_paths))

        max_rt_path_index = np.argmax(paths_tp1_sum_rt)
        max_rt_tp1 = paths_tp1_sum_rt[max_rt_path_index]
        max_rt_path = parallel_paths[max_rt_path_index]
        node_name = 'P{}'.format(self.p_node_num)

        self.p_node_num += 1
        self.simple_dag.add_node(node_name, rt=max_rt_tp1, path=max_rt_path)
        self.simple_dag.add_weighted_edges_from([(max_rt_path[0], node_name, 1)])
        self.simple_dag.add_weighted_edges_from([(node_name, max_rt_path[-1], 1)])
        self.node_delay[node_name] = 0
        self.edge_delay[(max_rt_path[0], node_name)] = 0
        self.edge_delay[(node_name, max_rt_path[-1])] = 0
        self.remove_paths(parallel_paths)
        self.update_rt()


    def process_self_loops(self):
        processed = False
        self_loops = [item for item in nx.selfloop_edges(self.simple_dag, data=True)]
        self_loops = [(item[0], item[2]['weight'],
                       self.rt[item[0]] + self.node_delay[item[0]] + self.edge_delay[(item[0], item[0])]) for item in
                      self_loops]

        for self_loop in self_loops:
            processed = True
            out_edges = {item: self.simple_dag.get_edge_data(item[0], item[1])['weight'] for item in
                         self.simple_dag.out_edges(self_loop[0])}
            tp_list = [out_edges[item] for item in out_edges]

            if (round(np.sum(tp_list), 8) != 1.0):
                raise Exception(f'Invalid Self-loop: Node:{self_loop[0]}')

            new_rt = self_loop[2] / (1 - self_loop[1])
            self.simple_dag.nodes[self_loop[0]]['rt'] = new_rt
            self.simple_dag.remove_edge(self_loop[0], self_loop[0])
            del out_edges[(self_loop[0], self_loop[0])]
            tp_denominator = 1.0 - self_loop[1]

            for edge in out_edges:
                self.simple_dag[edge[0]][edge[1]]['weight'] = out_edges[edge] / tp_denominator

        self.update_rt()

        return processed


    def simplify_parallels(self):
        processed = False
        parallel_paths = self.get_parallel_paths_p1(self.simple_dag)

        for pp in parallel_paths:
            paths_with_b_node = [p for p in pp if
                                 len([node for node in p if type(node) == str and node[0] == 'B']) != 0]

            if (len(paths_with_b_node) != 0):
                pp_tp_1 = pp.copy()

                for path in paths_with_b_node:
                    pp_tp_1.remove(path)

                if len(pp_tp_1) > 1:
                    self.merge_parallel_paths(pp_tp_1)
                    processed = True
                    continue

                if (len(pp_tp_1) != 0):
                    pp_tp_1_rt = self.sum_rt(paths=pp_tp_1[0], include_start_node=False, include_end_node=False,
                                            include_first_edge_delay=True, include_last_edge_delay=True)
                else:
                    pp_tp_1_rt = None

                rttp = self.get_rttp_for_paths_with_b_node(paths_with_b_node)
                refined_rttp = self.process_rttp(rttp, pp_tp_1_rt)
                self.remove_paths(paths_with_b_node)

                if (len(pp_tp_1) != 0):
                    self.remove_paths(pp_tp_1[0])

                for pnode in refined_rttp:
                    node_name = 'P{}'.format(self.p_node_num)
                    self.p_node_num += 1
                    self.simple_dag.add_node(node_name, rt=pnode[0])
                    self.simple_dag.add_weighted_edges_from([(pp[0][0], node_name, pnode[1]), (node_name, pp[0][-1], 1)])
                    self.node_delay[node_name] = 0
                    self.edge_delay[(pp[0][0], node_name)] = 0
                    self.edge_delay[(node_name, pp[0][-1])] = 0
                    processed = True

                continue

            self.merge_parallel_paths(pp)

            processed = True

        self.update_rt()

        return processed


    def simplify_branches(self):
        processed = False

        if (self.is_simple()):
            return

        parallel_paths = self.get_parallel_paths(self.simple_dag)

        for pp in parallel_paths:
            out_degree_pr = sum(map(lambda p: self.simple_dag.get_edge_data(p[0], p[1])['weight'], pp))

            if (out_degree_pr == 1):
                paths_sum_rt = self.sum_rt(paths=pp, include_start_node=False, include_end_node=False,
                                          include_first_edge_delay=True, include_last_edge_delay=True)
                paths_tp = self.get_tp(self.simple_dag, pp)
                node_name = 'B{}'.format(self.b_node_num)
                self.b_node_num += 1
                self.simple_dag.add_node(node_name, rt_list=paths_sum_rt, tp_list=paths_tp, original_paths=pp,
                                        rt=np.sum(np.multiply(paths_sum_rt, paths_tp)))
                self.simple_dag.add_weighted_edges_from([(pp[0][0], node_name, 1), (node_name, pp[0][-1], 1)])
                self.node_delay[node_name] = 0
                self.edge_delay[(pp[0][0], node_name)] = 0
                self.edge_delay[(node_name, pp[0][-1])] = 0
                self.remove_paths(pp)
                processed = True
                continue

            path_tp1_filter = filter(lambda p: self.simple_dag.get_edge_data(p[0], p[1])['weight'] == 1, pp)

            try:
                path_tp1 = next(path_tp1_filter)
                path_tp1_rt = self.sum_rt(paths=path_tp1, include_start_node=False, include_end_node=False,
                                         include_first_edge_delay=True, include_last_edge_delay=True)

            except (StopIteration, KeyError):
                # StopIteration: no tp==1 path in this group (the intended case).
                # KeyError: a path referencing an already-removed node.
                # Previously a bare `except:`, which hid both — and everything else.
                continue

            pp.remove(path_tp1)
            paths_sum_rt = list(map(
                lambda x: self.sum_rt(paths=x, include_start_node=False, include_end_node=False,
                                         include_first_edge_delay=True, include_last_edge_delay=True), pp))
            paths_gte_path_tp1_rt_filter = filter(lambda x: paths_sum_rt[x] >= path_tp1_rt, range(0, len(pp)))
            paths_gte_path_tp1_rt_index = list(paths_gte_path_tp1_rt_filter)
            paths_tpn1_prob_sum = sum(
                [self.simple_dag.get_edge_data(pp[i][0], pp[i][1])['weight'] for i in paths_gte_path_tp1_rt_index])

            if (paths_tpn1_prob_sum > 1):
                raise Exception('The sum of probabilities is greater than 1. Paths:{}'.format(pp))

            self.simple_dag[path_tp1[0]][path_tp1[1]]['weight'] = 1.0 - paths_tpn1_prob_sum
            processed = True
            paths_lt_path_tp1_rt_index = list(set(range(0, len(pp))) - set(paths_gte_path_tp1_rt_index))

            for index in paths_lt_path_tp1_rt_index:
                self.remove_paths(pp[index])

        self.update_rt()

        return processed


    def simplify_loops(self):
        processed = False
        self.process_self_loops()
        cycles = self._graphs.discover_cycles(self.simple_dag, self.start_point)
        cycles_rttp = [((item[0], item[-1]), ((self.sum_rt(item, include_start_node=True, include_end_node=False,
                                                          include_first_edge_delay=False, include_last_edge_delay=True) +
                                               self.edge_delay[(item[-1], item[0])]), self.get_tp(self.simple_dag, item)))
                       for item in cycles]

        cycles_dict = {}

        for (key, rttp) in cycles_rttp:
            if (rttp[0] != False and cycles_dict.setdefault(key, {'tp_sum': 0, 'avg_rt': 0}) != False):
                cycles_dict[key]['tp_sum'] += rttp[1]
                if (round(cycles_dict[key]['tp_sum'], 4) != 1):
                    nodes_in_cycle = list(
                        set([node for path in nx.all_simple_paths(self.simple_dag, key[0], key[1]) for node in path]))

                    for node_in_cycle in nodes_in_cycle:
                        if node_in_cycle == key[0] or node_in_cycle == key[1]:
                            continue
                        out_nodes = set([edge[1] for edge in self.simple_dag.out_edges(node_in_cycle)])
                        if out_nodes & set(nodes_in_cycle) != out_nodes:
                            jump_nodes = list(out_nodes - (out_nodes & set(nodes_in_cycle)))

                            for jumpnode in jump_nodes:
                                out_edges = {item: self.simple_dag.get_edge_data(item[0], item[1])['weight'] for item in
                                             self.simple_dag.out_edges(node_in_cycle)}
                                tp_list = [out_edges[item] for item in out_edges]

                                if (round(np.sum(tp_list), 4) > 1.0):
                                    raise Exception(f'Invalid Node: Node:{key[1]}')

                                tp_denominator = 1.0 - self.simple_dag.get_edge_data(node_in_cycle, jumpnode)['weight']
                                self.simple_dag.remove_edge(node_in_cycle, jumpnode)
                                del out_edges[(node_in_cycle, jumpnode)]

                                for edge in out_edges:
                                    self.simple_dag[edge[0]][edge[1]]['weight'] = out_edges[edge] / tp_denominator

                                if nx.shortest_path_length(self.simple_dag, source='Start',
                                                           target=jumpnode) > nx.shortest_path_length(self.simple_dag,
                                                                                                      source='Start',
                                                                                                      target=key[1]):
                                    self.approximations.append(
                                        {'cycle': key, 'type': 'overestimated', 'jump_from': node_in_cycle,
                                         'jump_to': jumpnode, 'tp': 1 - tp_denominator})

                                elif nx.shortest_path_length(self.simple_dag, source='Start',
                                                             target=jumpnode) < nx.shortest_path_length(self.simple_dag,
                                                                                                        source='Start',
                                                                                                        target=key[1]):
                                    self.approximations.append(
                                        {'cycle': key, 'type': 'underestimated', 'jump_from': node_in_cycle,
                                         'jump_to': jumpnode, 'tp': 1 - tp_denominator})

                cycles_dict[key]['avg_rt'] += rttp[0] * rttp[1]

            else:
                cycles_dict[key] = False
        cycles_dict = {key: cycles_dict[key]['avg_rt'] for key in cycles_dict if
                       cycles_dict[key] != False and round(cycles_dict[key]['tp_sum'], 4) == 1}

        for key in cycles_dict:
            loop_tp = self.simple_dag.get_edge_data(key[1], key[0])['weight']
            out_edges = {item: self.simple_dag.get_edge_data(item[0], item[1])['weight'] for item in
                         self.simple_dag.out_edges(key[1])}
            tp_list = [out_edges[item] for item in out_edges]

            if (round(np.sum(tp_list), 4) > 1.0):
                raise Exception(f'Invalid Loop: Node:{key[1]}')

            new_rt = (self.dag_rt[key[1]] + loop_tp * cycles_dict[key]) / (1 - loop_tp)
            self.simple_dag.nodes[key[1]]['rt'] = new_rt
            self.simple_dag.remove_edge(key[1], key[0])
            processed = True
            del out_edges[(key[1], key[0])]
            tp_denominator = 1.0 - loop_tp

            for edge in out_edges:
                self.simple_dag[edge[0]][edge[1]]['weight'] = out_edges[edge] / tp_denominator

        self.update_rt()

        return processed
