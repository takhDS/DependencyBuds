import networkx as nx
import numpy as np
import pandas as pd
def parent_node_number(edges_array, nodes_array, includeIsolatedNodes: bool = True):
    G = nx.DiGraph()
    G.add_edges_from(edges_array)
    if includeIsolatedNodes:
        G.add_nodes_from(nodes_array)
    infected_nodes = [i for i in G.nodes() if G.in_degree(i) == 0]
    return infected_nodes

