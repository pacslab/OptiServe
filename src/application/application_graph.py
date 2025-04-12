import networkx as nx
from typing import Dict, List, Optional
from src.application.function import Function


class ApplicationGraph:
    def __init__(self, start_node: str = "Start", end_node: str = "End") -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.functions: Dict[str, Function] = {}  # Map function name to function object
        self.start_node: str = start_node
        self.end_node: str = end_node

    def add_function(self, func: Function) -> None:
        """Add a function (node) to the application."""
        self.functions[func.name] = func
        self.graph.add_node(func.name, function=func)

    def add_edge(
        self,
        from_func: Function,
        to_func: Function,
        delay: int = 0,
        probability: float = 1.0,
    ) -> None:
        """
        Add an edge representing the interaction/transition between two functions.
        The edge can store additional metadata such as delay and transition probability.
        """
        self.graph.add_edge(from_func.name, to_func.name, delay=delay, prob=probability)

    def update_function_attributes(self) -> None:
        """
        Synchronize the node attributes in the graph with your function objects.
        For example, if a function's chosen memory configuration or response time has been updated
        (perhaps by your performance modeling code), you can update the graph.
        """
        for name, func in self.functions.items():
            if name in self.graph.nodes:
                self.graph.nodes[name]["function"] = func

    def compute_path_weight(self, path: List[str]) -> float:
        """
        Compute the weight (or criticality) of a given simple path.
        Here you can combine the per-function response time, possibly multiplied by a factor
        derived from the model type, and also include edge delays and probabilities.
        """
        total_weight: float = 0.0
        transition_product: float = 1.0
        for idx, node_name in enumerate(path):
            func: Function = self.functions[node_name]

            weight_factor: float = func.weight_factor
            total_weight += weight_factor
            if idx < len(path) - 1:
                edge_data = self.graph.get_edge_data(path[idx], path[idx + 1])
                p: float = edge_data.get("prob", 1.0)
                transition_product *= p
                total_weight += edge_data.get("delay", 0)

        return total_weight * transition_product

    def find_critical_path(self) -> Optional[List[str]]:
        """
        Find the most “critical” (or heavy) path from start_node to end_node.
        """
        all_paths: List[List[str]] = list(
            nx.all_simple_paths(
                self.graph, source=self.start_node, target=self.end_node
            )
        )
        if not all_paths:
            return None

        critical_path: List[str] = max(all_paths, key=self.compute_path_weight)
        return critical_path

    def get_sorted_paths(self, reverse: bool = True) -> List[tuple[List[str], float]]:
        """
        Retrieve all simple paths from start_node to end_node sorted based on criticality.
        By default, the sort is in descending order (highest weight first).

        Args:
            reverse (bool): If True, sort in descending order of computed weight.
        """
        all_paths: List[List[str]] = list(
            nx.all_simple_paths(
                self.graph, source=self.start_node, target=self.end_node
            )
        )
        weighted_paths: List[tuple[List[str], float]] = [
            (path, self.compute_path_weight(path)) for path in all_paths
        ]
        weighted_paths.sort(key=lambda x: x[1], reverse=reverse)

        return weighted_paths
