import networkx as nx
import matplotlib.pyplot as plt


from typing import Dict
from collections import defaultdict


def get_app1_graph(mem_config_list: Dict[str, float] = defaultdict(lambda: 0),
                          function_duration_dict: Dict[str, float] = defaultdict(lambda: 0)):
    app1_graph = nx.DiGraph()
    app1_graph.add_node('Start', pos=(0, 1))
    app1_graph.add_node(1, pos=(1, 1), mem = mem_config_list['F1'], rt = function_duration_dict['F1'])
    app1_graph.add_node(2, pos=(2, 2), mem = mem_config_list['F2'], rt = function_duration_dict['F2'])
    app1_graph.add_node(3, pos=(2, 0), mem = mem_config_list['F3'], rt = function_duration_dict['F3'])
    app1_graph.add_node(4, pos=(3, 2.5), mem = mem_config_list['F4'], rt = function_duration_dict['F4'])
    app1_graph.add_node(5, pos=(3, 1.5), mem = mem_config_list['F5'], rt = function_duration_dict['F5'])
    app1_graph.add_node(6, pos=(3, 0), mem = mem_config_list['F6'], rt = function_duration_dict['F6'])
    app1_graph.add_node(7, pos=(4, 1), mem = mem_config_list['F7'], rt = function_duration_dict['F7'])
    app1_graph.add_node(8, pos=(5, 1), mem = mem_config_list['F8'], rt = function_duration_dict['F8'])
    app1_graph.add_node('End', pos=(6, 1))
    app1_graph.add_weighted_edges_from([(1, 2, 1),(1, 3, 1),(2, 4, 0.6),(2, 5, 0.4),(4, 7, 1),(5, 7, 1),(3, 6, 1), (6, 7, 0.9), (6, 3, 0.1), (7, 8, 0.8),(7, 7, 0.2)])
    app1_graph.add_weighted_edges_from([('Start', 1, 1), (8, 'End', 1)])
    pos_app1_graph = nx.get_node_attributes(app1_graph, 'pos')
    labels_app1_graph = nx.get_edge_attributes(app1_graph, 'weight')
    
    nx.draw(app1_graph, pos_app1_graph, with_labels=True)
    nx.draw_networkx_edge_labels(app1_graph, pos_app1_graph, edge_labels=labels_app1_graph)
    pos_higher_offset_app1_graph = {}
    
    for k, v in pos_app1_graph.items():
        pos_higher_offset_app1_graph[k] = (v[0], v[1] + 0.15)

    plt.savefig('app1_graph.png')
    
    return app1_graph