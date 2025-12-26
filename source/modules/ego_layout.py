import networkx as nx
import pandas as pd 
import numpy as np
import random
from math import cos, sin, pi
def ego_position(edges_array, nodes_array, init_infected: int = None, jitter: float = 0.15, radial_jitter: float = 0.25):
    """Creates an ego graph and sets positions for visualization in a spreading manner -> Since these are so small comparatively to the full graph, singletons are not used

    Parameters
    ----------
    edges_array: list
        Array of all edges in network.
    nodes_array: list
        Array of all nodes in network (includes isolated nodes).
    init_infected: int
        The number of the initially infected node in the network, if set to None a random node is picked.
    jitter: float
        The amount of jitter within the circle, to attempt to visualize each node better.
    radial_jitter: float
        The amount of jitter of the actual circle itself, to attempt to visualize each node better.
    Returns
    -------
    G: nx.ego_graph()
        The graph G containing the ego network for node: init_infected
    pos: dict
        A dictionary containing the position of each node for visualization
    init_infected: 
        The initially infected node that needs to be passed through in the SIR simulation later.
    """
    # We have to select a node at this stage in order to create the ego network -> can also be set from a parameter
    if init_infected == None:
        init_infected = np.random.choice(nodes_array, 1)[0]
    # Creating graph with ALL NODES (including isolated ones we used to miss) and adding the isolated nodes as a number for the graph (will not show up in ego network so might be redundant here)
    G = nx.DiGraph()
    G.add_edges_from(edges_array)
    G.add_nodes_from(nodes_array)
    isolatedNodes = list(nx.isolates(G))
    G.remove_nodes_from(isolatedNodes)
    # Adding this one back in case it stems from the isolated nodes (especially when using random choice)
    G.add_node(init_infected)
    # Creating the actual ego graph
    G = nx.ego_graph(G, n=init_infected, radius=999)

    # Creating a distance based positioning to make nodes closer to ego node (i.e. connected to ego, vs 1 away, 2 away etc.) more centered.
    distances = nx.shortest_path_length(G, source=init_infected)
    
    # Keeping track of distances per node
    dist_dict = {}
    for node, dist in distances.items():
        dist_dict.setdefault(dist, []).append(node)
    
    # Position dict to keep track of all the positions
    pos = {}
    # Space between circles
    radius = 3.5
    # Placing the nodes according to their distance
    for dist, nodes in dist_dict.items():
        # Places ego node in center
        if dist == 0:
            pos[init_infected] = (0, 0)
            continue
        
        # Computes the radius for nodes based on distance, if one away then 1*radius, two away, then 2*radius etc.
        r = dist * radius
        # Places the nodes next to each other in a circular fashion
        angle_step = 2 * pi / len(nodes)

        # Sets the position of each node based on angle and jittered radius
        for i, node in enumerate(nodes):
            angle = i * angle_step + random.uniform(-jitter, jitter)
            jittered_r = r + random.uniform(-radial_jitter, radial_jitter)
            pos[node] = (jittered_r * cos(angle), jittered_r * sin(angle))


    return G, pos, init_infected
