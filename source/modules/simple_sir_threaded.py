import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import copy
from multiprocessing import Pool, cpu_count

init_susceptible_color = np.array([0.1, 1, 0.1, 1])
susceptible_color = np.array([0.1, 1, 0.1, 1])
infected_color = np.array([1, 0.1, 0.1, 1])
removed_color = np.array([0.1, 0.1, 1, 1])
quarantined_color = np.array([0.5, 0.5, 0.5, 1])

def sir_model(G, 
              pos, 
              init_infected: int = None, 
              max_steps: int = 100, 
              infection_rate: float = 0.2, 
              recovery_rate: float = 0.05, 
              noticeability_rates: float = None, # A tuple, where first value is before virus found, second value after virus found -- increased awareness!
              doVisualization: bool = True, 
              doSingletonReduction: bool = True, 
              doVisualizeSingletonReduction: bool = True,
              doRenderInfoText: bool = True):
    """
    Simulates SIR model and returns visualization information and singleton reduction if needed.
    """
    if not doVisualization:
        doRenderInfoText = False
        doVisualizeSingletonReduction = False

    # Stores the states for each step in the simulation
    nodelist_total = []
    nodecolors_total = []
    edgecolors_total = []
    infotext_total = [] # Tuples of form: (susceptible, infected, recovered, quarantined)
    singletons_total = []
    #if none is chosen to be targeted, choose random
    if init_infected is None:
        init_infected = np.random.choice(list(G.nodes), 1)[0]
    # I need to create an ego graph in order to only grab susceptible singletons within the network
    if init_infected:
        G_2 = copy.deepcopy(G)
        G_2 = nx.ego_graph(G, n=init_infected, radius=9999)
    # Ensure state is initialized (starts at 0)
    if nx.get_node_attributes(G, 'state') == {}:
        nx.set_node_attributes(G, 0, 'state')

    # If none is chosen to be targeted, choose random
    if init_infected is None:
        init_infected = np.random.choice(list(G.nodes), 1)[0]
    
    # infects the random one
    G.nodes[init_infected]['state'] = 1

    # Visualization options
    options = {"node_size": 20}
    global init_susceptible_color
    global susceptible_color
    global infected_color
    global removed_color
    global quarantined_color

    def get_singleton_edgecolor(node_num, quarantined_list):
        """
        Under each node, there is an attribute for each state every child singleton can be in, and how many singletons are in.
        The following function gets the average color of all the singletons for the purpose of rendering the edge color of the
        parent node.
        """
        if node_num in quarantined_list:
            return quarantined_color
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

    # Using sets to keep track of infected nodes
    infected_list = set()
    for i, state in G.nodes.data('state'):
        if state == 1:
            infected_list.add(i)
    
    # Keeps track of whether or not a node has singletons infected
    has_infected_singletons = set()
    if doSingletonReduction:
        for i, inf_sng in G.nodes.data('i_singletons'):
            if inf_sng > 0:
                has_infected_singletons.add(i)

    # Keep track of origin nodes of quarantine - levels of contact tracing 'spread' from them
    quarantined_origin_list = set() # Nodes that are within quarantine don't get infected and don't spread, singletons don't get quarantined
    virusFound = noticeability_rates == None

    if doVisualization: # Saves the colors of the nodes
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
                node_colors.append(init_susceptible_color)
            edge_colors.append(get_singleton_edgecolor(n, quarantined_list=set()))

        nodelist_total.append(nodelist)
        nodecolors_total.append(node_colors)
        edgecolors_total.append(edge_colors)


    # Appending the initial state of the singletons to a list for visualization purposes
    susceptible_singletons_outside = sum(x[1] for x in G.nodes.data('s_singletons')) if doSingletonReduction else 0
    susceptible_singletons_within = sum(x[1] for x in G_2.nodes.data('s_singletons')) if doSingletonReduction else 0
    susceptible_singletons_outside = susceptible_singletons_outside - susceptible_singletons_within
    infected_singletons = sum(x[1] for x in G.nodes.data('i_singletons')) if doSingletonReduction else 0
    recovered_singletons = sum(x[1] for x in G.nodes.data('r_singletons')) if doSingletonReduction else 0
    singletons_total.append((susceptible_singletons_within, infected_singletons, recovered_singletons, susceptible_singletons_outside))




    print("Running model...")

    total_susceptible_singletons = sum(x[1] for x in G.nodes.data('s_singletons')) if doSingletonReduction else 0
    total_susceptible = len(G) + total_susceptible_singletons - len(infected_list)
    total_recovered = 0

    # Run model
    for step in range(max_steps):
        nodes_to_draw = set(sorted(list(G.nodes())))
        # Appending the state of singletons for every single step the model uses
        susceptible_singletons_within = sum(x[1] for x in G_2.nodes.data('s_singletons')) if doSingletonReduction else 0
        infected_singletons = sum(x[1] for x in G.nodes.data('i_singletons')) if doSingletonReduction else 0
        recovered_singletons = sum(x[1] for x in G.nodes.data('r_singletons')) if doSingletonReduction else 0
        susceptible_singletons_within = susceptible_singletons_within - infected_singletons - recovered_singletons
        singletons_total.append((susceptible_singletons_within, infected_singletons, recovered_singletons, susceptible_singletons_outside))
        infected_non_singletons = len(infected_list)
        total_infected = infected_non_singletons + infected_singletons

        quarantined_list = set()
        for origin in quarantined_origin_list:
            quarantined_list.add(origin)
            adj = G.neighbors(origin)
            for neighbor in adj:
                quarantined_list.add(neighbor)

        if virusFound:
            noticeability_rate = noticeability_rates[1]
        else:
            noticeability_rate = noticeability_rates[0]

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Step: {step}")
        print(f"Infected left: {total_infected}")
        print(f"> Non-singletons infected: {infected_non_singletons}")
        print(f"> Singletons infected: {infected_singletons}")
        
        # Save info about network as output
        if doRenderInfoText:
            infotext_total.append((total_susceptible, total_infected, total_recovered, 0)) 

        # Early Stopping
        if infected_non_singletons == 0 and infected_singletons == 0:
            break

        # >>> Infection, recovery and noticeability
        if doSingletonReduction and virusFound:
            print("Recovering singletons...")
            temp_infsng_list = list(has_infected_singletons) # Temporary infected singleton list
            for i in temp_infsng_list:
                inf_sng = G.nodes[i]['i_singletons']
                if inf_sng > 0:
                    temp = np.random.binomial(n=inf_sng, p=recovery_rate)
                    if temp > 0:
                        G.nodes[i]['i_singletons'] -= temp
                        G.nodes[i]['r_singletons'] += temp
                        total_recovered += temp
                        if doVisualization:
                            nodes_to_draw.add(i)

                    # > Noticeability and quarantine
                    # temp_noticed = np.random.binomial(n=temp, p=noticeability_rate)
                    # if temp_noticed > 0:
                else:
                    has_infected_singletons.remove(i)

        if doSingletonReduction:
            print("Infecting singletons and neighbors + Recovering infected nodes...")
        else:
            print("Infecting neighbors + Recovering infected nodes...")
            
        temp_inf_list = list(infected_list)
        for i in temp_inf_list:
            # Quarantine current infected node
            if np.random.sample() < noticeability_rate and not i in quarantined_list:
                # print(f"Noticed {i}!")
                quarantined_origin_list.add(i)
                quarantined_list.add(i)
                virusFound = True

            isQuarantined = i in quarantined_list
            # Infect adjacent nodes
            if not isQuarantined:
                adj = G.neighbors(i)
                for j in adj:
                    if G.nodes[j].get('state', 0) == 0 and np.random.sample() < infection_rate:
                        infected_list.add(j)
                        G.nodes[j]['state'] = 1
                        total_susceptible -= 1
                        if doVisualization:
                            nodes_to_draw.add(j)

            # Infect singletons
            if doSingletonReduction and not isQuarantined:
                temp = np.random.binomial(n=G.nodes[i]['s_singletons'], p=infection_rate)
                if temp > 0:
                    has_infected_singletons.add(i)
                    G.nodes[i]['s_singletons'] -= temp
                    G.nodes[i]['i_singletons'] += temp
                    total_susceptible -= temp
                    if doVisualization:
                        nodes_to_draw.add(i)

            # Recover infected nodes
            if np.random.sample() < recovery_rate and virusFound:
                infected_list.remove(i)
                G.nodes[i]['state'] = 2
                total_recovered += 1
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
                else: node_colors.append(init_susceptible_color)
                edge_colors.append(get_singleton_edgecolor(n, quarantined_list))
            
            nodelist_total.append(nodelist)
            nodecolors_total.append(node_colors)
            edgecolors_total.append(edge_colors)
    if doSingletonReduction:
        return nodelist_total, nodecolors_total, edgecolors_total, options, singletons_total
    else:
        return nodelist_total, nodecolors_total, edgecolors_total, options
    
def work(i, G, pos, nodelist_total, nodecolors_total, edgecolors_total, options, visualizeEdges: bool = False, isolatedNodes: list = None, init_infected: int = None, singletons_total: list = None):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    
    print(f"Starting graph image: {i}")
    fig, ax = plt.subplots()
    if init_infected:
        G_2 = copy.deepcopy(G)
        G_2 = nx.ego_graph(G, n=init_infected, radius=9999)
        isolatedNew = isolatedNodes + (len(G.nodes()) - len(G_2.nodes()))
    
    global susceptible_color
    global infected_color
    global removed_color
    global quarantined_color
    
    # Define SIR colors (RGBA)
    state_to_color = {
        0: susceptible_color,  # Susceptible
        1: infected_color,  # Infected
        2: removed_color   # Removed
    }
    # plt.Circle((5, 5), 0.5, color='b', fill=False)

    # Group nodes by state
    state_to_nodes = {0: [], 1: [], 2: []}
    state_to_edgecolors = {0: [], 1: [] , 2: []}

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
            

    plt.text(0, 1, "Susceptible: " + str(infotext_total[i][0]) + '\n' + 
                      "Infected: " + str(infotext_total[i][1]) + '\n' + 
                      "Recovered: " + str(infotext_total[i][2]) + '\n', 
                      horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    ax.axis("off")
    fig.text(-0.15, 0.95, s=f"Step: {i+1}")
    # Printing the numbers including the singletons if they are present
    if singletons_total:
        fig.text(-0.15, 0.9, s=f"Total Nodes = {len(G.nodes()) + isolatedNodes + singletons_total[i][0] + singletons_total[i][1] + singletons_total[i][2] + singletons_total[i][3]}")
        fig.text(-0.15, 0.85, s=f"Susceptible Nodes: {len(state_to_nodes[0]) - (len(G.nodes()) - len(G_2.nodes())) + singletons_total[i][0]}")
        fig.text(-0.15, 0.8, s=f"Infected Nodes: {len(state_to_nodes[1]) + singletons_total[i][1]}")
        fig.text(-0.15, 0.75, s=f"Recovered Nodes: {len(state_to_nodes[2]) + singletons_total[i][2]}")
        if isolatedNodes:
            if not init_infected:
                fig.text(-0.15, 0.7, s=f"Isolated Nodes: {isolatedNodes}")
            if init_infected:
                fig.text(-0.15, 0.7, s=f"Isolated Nodes: {isolatedNew + singletons_total[i][3]}")
    # Printing the numbers without the singletons if they are not present
    else:
        fig.text(-0.15, 0.9, s=f"Total Nodes = {len(G.nodes()) + isolatedNodes}")
        fig.text(-0.15, 0.85, s=f"Susceptible Nodes: {len(state_to_nodes[0]) - (len(G.nodes()) - len(G_2.nodes()))}")
        fig.text(-0.15, 0.8, s=f"Infected Nodes: {len(state_to_nodes[1])}")
        fig.text(-0.15, 0.75, s=f"Recovered Nodes: {len(state_to_nodes[2])}")
        if isolatedNodes:
            if not init_infected:
                fig.text(-0.15, 0.7, s=f"Isolated Nodes: {isolatedNodes}")
            if init_infected:
                fig.text(-0.15, 0.7, s=f"Isolated Nodes: {isolatedNew}")
    print(singletons_total[i])
    fig.subplots_adjust(left=0.2) 
    fig.tight_layout()
    fig.savefig(f"graphs/graph_{i}.png", format="PNG", bbox_inches='tight')
    print(f"Saved graph_{i}.png!")
    plt.close(fig)

