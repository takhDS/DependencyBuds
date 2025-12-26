import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import copy
from multiprocessing import Pool, cpu_count

def sir_model(G, pos, init_infected: int = None, max_steps: int = 100, infection_rate: float = 0.2, recovery_rate: float = 0.05, doVisualization: bool = True, doSingletonReduction: bool = True, doVisualizeSingletonReduction: bool = True):
    """Creates SIR model and visualizes it if enabled
    
    """
    #stores the states for each step in the simulation 
    nodelist_total = []
    nodecolors_total = []
    edgecolors_total = []

    #if none is chosen to be targeted, choose random
    if init_infected is None:
        init_infected = np.random.choice(list(G.nodes), 1)[0]
    
    # Ensure state is initialized (starts at 0)
    if nx.get_node_attributes(G, 'state') == {}:
        nx.set_node_attributes(G, 0, 'state')

    
    #infects the random one
    G.nodes[init_infected]['state'] = 1

    # Visualization options
    options = {"node_size": 20}
    init_susceptible_color = np.array([0, 1, 0, 0.5])
    susceptible_color = np.array([0, 1, 0, 1])
    infected_color = np.array([1, 0, 0, 1])
    removed_color = np.array([0, 0, 1, 1])

    '''singelton is an attribute under each node, for the number of singeltons pointing to a node, and then assigns 
    a value of s = to the number of singelton pointing to the parent node, then the following function gets the color
    for each singelton edge'''
    def get_singleton_edgecolor(node_num):
        x = G.nodes[node_num]
        if not doSingletonReduction or not doVisualizeSingletonReduction:
            st = x.get('state', 0)
            if st == 0: return susceptible_color
            if st == 1: return infected_color
            return removed_color
        
        singleton_sum = x.get('s_singletons', 0) + x.get('i_singletons', 0) + x.get('r_singletons', 0)
        if singleton_sum == 0:
            st = x.get('state', 0)
            if st == 0: return susceptible_color
            if st == 1: return infected_color
            return removed_color
        else:
            temp_color = x['s_singletons']*susceptible_color + x['i_singletons']*infected_color + x['r_singletons']*removed_color
            return temp_color / singleton_sum

    # if doVisualization:
    #     # Create graphs directory
    #     if not os.path.exists("graphs"):
    #         os.makedirs("graphs")
            
    #     # Draw network edges
    #     print("Drawing network edges...")
    #     nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.25)

    #     # Draw network nodes initially
    #     s_nodes = [x[0] for x in G.nodes.data('state') if x[1] == 0]
    #     i_nodes = [x[0] for x in G.nodes.data('state') if x[1] == 1]
        
    #     if s_nodes:
    #         nx.draw_networkx_nodes(G, pos, nodelist=s_nodes, alpha=0.45, node_color=[init_susceptible_color], **options, edgecolors=[susceptible_color])
        
    #     if i_nodes:
    #         e_color = susceptible_color if doVisualizeSingletonReduction else infected_color
    #         nx.draw_networkx_nodes(G, pos, nodelist=i_nodes, node_color=[infected_color], **options, edgecolors=[e_color])

    #     print("Done!")

    #     # Export init graph
    #     plt.axis("off")
    #     plt.tight_layout()
    #     print("Saving graph_0.png...")
    #     plt.savefig("graphs/graph_0.png", format="PNG")
    #     print("Done!")

    # Using sets to keep track of infected nodes
    infected_list = set()
    for i, state in G.nodes.data('state'):
        if state == 1:
            infected_list.add(i)
    
    #keeps track of if a node has singeltons affecred
    has_infected_singletons = set()
    if doSingletonReduction:
        for i, inf_sng in G.nodes.data('i_singletons'):
            if inf_sng > 0:
                has_infected_singletons.add(i)

    # Run model
    if doVisualization: #saves the colors of the nodes
        nodelist = list(G.nodes())
        node_colors = []
        edge_colors = []
        for n in nodelist:
            st = G.nodes[n].get('state', 0)  # 0 = susceptible by default
            if st == 1:
                node_colors.append(infected_color)
            elif st == 2:
                node_colors.append(removed_color)
            else:
                node_colors.append(susceptible_color)
            edge_colors.append(get_singleton_edgecolor(n))

        nodelist_total.append(nodelist)
        nodecolors_total.append(node_colors)
        edgecolors_total.append(edge_colors)
    print("Running model...")
    for step in range(max_steps):
        nodes_to_draw = set(sorted(list(G.nodes())))
        
        infected_singletons = sum(x[1] for x in G.nodes.data('i_singletons')) if doSingletonReduction else 0
        infected_non_singletons = len(infected_list)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Step: {step}")
        print(f"Infected left: {infected_non_singletons + infected_singletons}")
        print(f"> Non-singletons infected: {infected_non_singletons}")
        print(f"> Singletons infected: {infected_singletons}")

        # Early Stopping
        if infected_non_singletons == 0 and infected_singletons == 0:
            break

        # >>> Infection and recovery
        if doSingletonReduction:
            print("Recovering singletons...")
            temp_infsng_list = list(has_infected_singletons) #temporary infected singelton list
            for i in temp_infsng_list:
                inf_sng = G.nodes[i]['i_singletons']
                if inf_sng > 0:
                    temp = np.random.binomial(n=inf_sng, p=recovery_rate)
                    if temp > 0:
                        G.nodes[i]['i_singletons'] -= temp
                        G.nodes[i]['r_singletons'] += temp
                        if doVisualization:
                            nodes_to_draw.add(i)
                else:
                    has_infected_singletons.remove(i)

        if doSingletonReduction:
            print("Infecting singletons and neighbors + Recovering infected nodes...")
        else:
            print("Infecting neighbors + Recovering infected nodes...")
            
        temp_inf_list = list(infected_list)
        for i in temp_inf_list:
            # Infect adjacent nodes
            adj = G.neighbors(i)
            for j in adj:
                if G.nodes[j].get('state', 0) == 0 and np.random.sample() < infection_rate:
                    infected_list.add(j)
                    G.nodes[j]['state'] = 1
                    if doVisualization:
                        nodes_to_draw.add(j)

            # Infect singletons
            if doSingletonReduction:
                temp = np.random.binomial(n=G.nodes[i]['s_singletons'], p=infection_rate)
                if temp > 0:
                    has_infected_singletons.add(i)
                    G.nodes[i]['s_singletons'] -= temp
                    G.nodes[i]['i_singletons'] += temp
                    if doVisualization:
                        nodes_to_draw.add(i)

            # Recover infected nodes
            if np.random.sample() < recovery_rate:
                infected_list.remove(i)
                G.nodes[i]['state'] = 2
                if doVisualization:
                    nodes_to_draw.add(i)

        # Draw all changes for this step in one go
        if doVisualization and nodes_to_draw:
            nodelist = list(nodes_to_draw)
            node_colors = []
            edge_colors = []
            for n in nodelist:
                st = G.nodes[n].get('state', 0)
                if st == 1: node_colors.append(infected_color)
                elif st == 2: node_colors.append(removed_color)
                else: node_colors.append(susceptible_color)
                edge_colors.append(get_singleton_edgecolor(n))
            
            nodelist_total.append(nodelist)
            nodecolors_total.append(node_colors)
            edgecolors_total.append(edge_colors)
    return nodelist_total, nodecolors_total, edgecolors_total, options     
    
def work(i, G, pos, nodelist_total, nodecolors_total, edgecolors_total, options, visualizeEdges: bool = False, isolatedNodes: list = None, init_infected: int = None):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    print(f"Starting graph image: {i}")
    fig, ax = plt.subplots()
    if init_infected:
        G_2 = copy.deepcopy(G)
        G_2 = nx.ego_graph(G, n=init_infected, radius=9999)
        isolatedNew = isolatedNodes + (len(G.nodes()) - len(G_2.nodes()))
    # Define SIR colors (RGBA)
    state_to_color = {
        0: np.array([0, 1, 0, 1]),  # Susceptible
        1: np.array([1, 0, 0, 1]),  # Infected
        2: np.array([0, 0, 1, 1])   # Removed
    }

    # Group nodes by state
    state_to_nodes = {0: [], 1: [], 2: []}
    state_to_edgecolors = {0: [], 1: [], 2: []}

    for node, color, edge_color in zip(nodelist_total[i], nodecolors_total[i], edgecolors_total[i]):
        # Match color to state
        matched_state = None
        for s, c in state_to_color.items():
            if np.allclose(color, c):
                matched_state = s
                break
        if matched_state is None:
            raise ValueError(f"Unknown node color: {color}")
        state_to_nodes[matched_state].append(node)
        state_to_edgecolors[matched_state].append(edge_color)

    # Draw nodes in order: Susceptible → Removed → Infected (infected on top)
    layering_order = [0, 1, 2]
    for state in layering_order:
        nodes = state_to_nodes[state]
        if nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=state_to_color[state],
                edgecolors=state_to_edgecolors[state],
                **options
            )
        if visualizeEdges:
            nx.draw_networkx_edges(
                G, pos,
                alpha=0.18,
                width=0.7,
                edge_color="gray",
                connectionstyle="arc3,rad=0.1"
            )


    ax.axis("off")
    fig.text(-0.15, 0.95, s=f"Step: {i+1}")
    fig.text(-0.15, 0.9, s=f"Total Nodes = {len(G.nodes()) + isolatedNodes}")
    fig.text(-0.15, 0.85, s=f"Susceptible Nodes: {len(state_to_nodes[0]) - (len(G.nodes()) - len(G_2.nodes()))}")
    fig.text(-0.15, 0.8, s=f"Infected Nodes: {len(state_to_nodes[1])}")
    fig.text(-0.15, 0.75, s=f"Recovered Nodes: {len(state_to_nodes[2])}")
    if isolatedNodes:
        if not init_infected:
            fig.text(-0.15, 0.7, s=f"Isolated Nodes: {isolatedNodes}")
        if init_infected:
            fig.text(-0.15, 0.7, s=f"Isolated Nodes: {isolatedNew}")
    fig.subplots_adjust(left=0.2) 
    fig.tight_layout()
    fig.savefig(f"graphs/graph_{i}.png", format="PNG", bbox_inches='tight')
    print(f"Saved graph_{i}.png!")
    plt.close(fig)

