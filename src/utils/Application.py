import uuid
import string
import random
import networkx as nx
from typing import List

from .function import Function


class Application:
    def __init__(self,
                 name=f'Application_{"".join(random.choices(string.ascii_letters, k=8))}',
                 functions: List[Function]=None,
                 edges: List[tuple]=[]):
        self.id = uuid.uuid4().hex
        self.graph = nx.DiGraph()
        self.name = name
        
        self.graph.add_nodes_from(functions)
        self.graph.add_edges_from(edges)
    
 
    def add_function(self, function: Function, edges: List[tuple]=[]):
        self.graph.add_node(function)
        self.graph.add_edges_from(edges)