import networkx as nx
from typing import List

from .Function import Function


class Application:
    def __init__(self, name, version, functions: List[Function]=None):
        self.graph = nx.DiGraph()
        self.name = name
        self.version = version
        
        for function in functions or []:
            self.graph.add_node(function)
    
 
    def add_function(self, function: Function):
        self.add_node(function)