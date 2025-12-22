import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
# Force non-interactive backend to prevent GUI crashes during threading
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor

def singleton_reduction(edges_array, doSingletonReduction: bool = True, doVisualization: bool = True):
    """Prepares graph and reduces graph nodes with in_degree 1 and out_degree 0 into singletons. 
    
    Parameters
    ----------
    doSingletonReduction : bool
        Decides whether or not singleton reduction should be applied
    doVisualization : bool
        Decides whether or not there should be a visualization when creating the simulation
    
    Returns
    -------
    G : nx.DiGraph
        A directed graph containing singletons if enabled, otherwise containing no singletons
    pos : dict
        A layout of all the nodes for visualization later -> note, this is only returned if doVisualization is set to true, and significantly slows down the runtime.
    """
 
    # Create the digraph
    G = nx.DiGraph()
    G.add_edges_from(edges_array)

    """    
    Singletons are nodes that have in-degree of 1 and out-degree of 0. 
    This means, in an SIR model, a singleton doesn't infect, but is only infected, then after can recover.
    Because of this, we can essentially reduce singletons to a number stored in its parent node, reducing the size of our network by 20%.
    """
    if doSingletonReduction:
        # Singleton information stored in parent.
        nx.set_node_attributes(G, 0, "s_singletons") # Susceptible singletons
        nx.set_node_attributes(G, 0, "i_singletons") # Infected singletons
        nx.set_node_attributes(G, 0, "r_singletons") # Recovered/removed singletons

        # Find and store singletons
        singletons = G.out_degree
        singletons = [x[0] for x in singletons if x[1] == 0]
        singletons = [x for x in singletons if G.in_degree(x) == 1]

        # Find nodes that are not singletons
        non_singletons = [x for x in list(G.nodes) if not x in singletons]

        # Store singletons as numbers
        for node in non_singletons:
            adj = G.neighbors(node)
            G.nodes[node]['s_singletons'] = len([x for x in adj if x in singletons])

        # Remove nodes that are singletons
        print("Node count before singleton reduction: ", len(G.nodes()))
        G.remove_nodes_from(singletons)

    if doVisualization:
        print("Calculating node layout and positions...")
        pos = nx.spring_layout(G, seed=8020, gravity=0.75)
        print("Done!")
    
    if doVisualization:
        return G, pos
    else:
        return G
