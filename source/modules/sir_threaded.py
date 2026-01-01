import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import copy
from multiprocessing import Pool, cpu_count

# Set the colors for simulation
init_susceptible_color = np.array([0.1, 1, 0.1, 1])
susceptible_color = np.array([0.1, 1, 0.1, 1])
infected_color = np.array([1, 0.1, 0.1, 1])
removed_color = np.array([0.1, 0.1, 1, 1])
quarantined_color = np.array([0.5, 0.5, 0.5, 1])

def get_accessible_sus_nodes(G, init_infected):
    # init_infected is a list of the initially infected nodes
    found = set()
    for search in init_infected:
        found.update(nx.descendants(G, search) | {search})
    return found

def sir_model(G, 
              G_full,
              init_infected: list = None, 
              max_steps: int = 100, 
              #infection_rate: float = 0.2, 
              infection_rate = 0.2,
              recovery_rate: float = 0.05, 
              noticeability_rates: tuple = None, 
              quarantine_length: int = 5,
              network_type: str = "full", 
              doVisualization: bool = True, 
              doSingletonReduction: bool = True,
              doVisualizeIsolates: bool = False,
              doVisualizeSingletonReduction: bool = True,
              doRenderInfoText: bool = True,
              ):
    """
    Simulates SIR model and returns visualization information and singleton reduction if needed.
    May perform quarantining, and render text.

    Parameters
    ----------
    G: graph
        A networkx graph object containing all observed nodes. It should 
    pos: idk
        Networkx object containing position information of nodes for visualization.
    init_infected: list
        A list of nodes to initially be infected. If an empty list, or None passed, a list containing a single random node from the observed nodes is chosen.
    max_steps: int
        Maximum number of steps the simulation takes if early stopping mechanism not activated.
    infection_rate: float
        Base/default value for the likelihood an infected node passes infection to an adjacent node.
    recovery_rate: float
        Base/default value for the likelihood an infetced node recovers from its infection and becomes removed.
    noitceability_rates: tuple
        A tuple of two values. Noticeability refers to the likelihood that an authority notices a virus is in a library.
        The first value represents the noticeability rate before the virus is found. The second value represents the rate after.
        Quarantining is not performed if set to None.
    quarantine_length: int
        For how many steps does the quarantine last. Set to None for infinite quarantines. Set to 0 to disable quarantining.
    network_type: str
        Takes on the values "full" or "ego". "full" means all nodes of the network have been used. "ego" refers to the fact an ego network was used.
    doVisualization: bool
        Whether or not visualization should be done.
    doSingletonReduction: bool
        Whether or not singleton reduction should be performed. Singletons are nodes with in-degree 1 and out-degree 0.
        Singletons can only be infected, but can't infect, thus through singleton reduction, they are represented as only numbers under their parent node or as an average edge color of a node.
    doVisualizeSingletonReduction: bool
        Whether or not to visualize singletons as their average color around their parent node. Otherwise, they aren't visualized at all.
    doRenderInfoText: bool
        Whether or not render simulation information as text in the output for visualization.

    Returns
    -------
    wip
    """
    if not doVisualization:
        doRenderInfoText = False
        doVisualizeSingletonReduction = False
        doVisualizeIsolates = False

    # Stores the states for each step in the simulation
    nodelist_total = []
    nodecolors_total = []
    edgecolors_total = []
    infotext_total = [] # A list of dictionaries, describing the state of the step in the simulation
    constants_total = {}
    
    # If none is chosen to be targeted, choose one random node
    if init_infected is None or len(init_infected) == 0:
        init_infected = list(np.random.choice(list(G.nodes), 1)[0])

    # Ensure state is initialized (starts at 0)
    if nx.get_node_attributes(G, 'state') == {}:
        nx.set_node_attributes(G, 0, 'state')
    
    # Infects the chosen random nodes
    for node in init_infected:
        G.nodes[node]['state'] = 1

    # Visualization options
    options = {"node_size": 20}
    global init_susceptible_color
    global susceptible_color
    global infected_color
    global removed_color
    global quarantined_color

    def get_singleton_edgecolor(node_num, quarantined_list=None):
        """
        Under each node, there is an attribute for each state every child singleton can be in, and how many singletons are in.
        The following function gets the average color of all the singletons for the purpose of rendering the edge color of the
        parent node.
        """
        if not quarantined_list == None and node_num in quarantined_list:
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
    quarantined_origin_list = dict() # Nodes that are within quarantine don't get infected and don't spread, singletons don't get quarantined
    virusFound = noticeability_rates == None # If no quarantining, then virus instantly noticed by authority

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


    # >>> Save initial state information
    average_out_degree = sum(dict(G.out_degree).values())/G.number_of_nodes() #average degree of the graph
    infection_islist = type(infection_rate) is list
    susceptible_total = len(G) - len(init_infected)
    accessible_sus_nodes = get_accessible_sus_nodes(G_full, init_infected)
    # > Constant
    # Total Nodes
    if network_type == "ego":
        constants_total['tn'] = len(G) + sum([x[1] for x in G.nodes.data('s_singletons')])
    elif network_type == "full":
        constants_total['tn'] = len(G_full) + sum([x[1] for x in G_full.nodes.data('s_singletons')])
    constants_total['xt'] = constants_total['tn'] - len(accessible_sus_nodes)

    # > Variable
    infotext = {}

    infotext['ssw'] = sum(i[1] for i in G_full.nodes.data('s_singletons') if i[0] in accessible_sus_nodes)
    constants_total['xt'] -= infotext['ssw']
    infotext['sso'] = sum([x[1] for x in G_full.nodes.data('s_singletons')]) - infotext['ssw']

    infotext['it'] = len(init_infected)
    infotext['st'] = len(accessible_sus_nodes) - len(init_infected) + infotext['ssw']
    infotext['rt'] = 0
    infotext['qt'] = 0
    if doSingletonReduction:
        infotext['ss'] = infotext['ssw'] + infotext['sso']
        infotext['is'] = 0
        infotext['rs'] = 0
    infotext_total.append(dict(infotext))

    """    
    'tn': total_nodes, (within and outside)
    'st': susceptible_total, (within)
    'it': infected_total,
    'rt': recovered_total,
    'qt': quarantined_total,
    'ssw': susceptible_singletons_within, ('within' means that these are the susceptible singletons accessible by a path from an init_infected)
    'sso': susceptible_singletons_outside, ('outside' means that they are not accessible by any path from any of the init_infected)
    'is': susceptible_singletons,
    'is': infected_singletons,
    'rs': recovered_singletons,
    'xt': isolated_total,
    """

    # Run model
    print("Running model...")
    for step in range(max_steps):
        nodes_to_draw = set(sorted(list(G.nodes())))

        # Update quarantined list
        quarantined_list = set()
        if not quarantine_length == 0:
            for origin in quarantined_origin_list.keys():
                quarantined_list.add(origin)
                adj = G.neighbors(origin)
                for neighbor in adj:
                    quarantined_list.add(neighbor)

        for x in list(quarantined_origin_list.items()):
            # Tick time on quarantines
            if not quarantine_length == None:
                quarantined_origin_list[x[0]] = quarantined_origin_list[x[0]] - 1
                if x[1] <= 0:
                    del quarantined_origin_list[x[0]]

        infotext['qt'] = len(quarantined_list)

        if not noticeability_rate is None:
            noticeability_rate = noticeability_rates[1] if virusFound else noticeability_rates[0]

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Step: ", step+1)
        # print("[DEBUG] SS: ", infotext['ss'])
        # print("[DEBUG] SSW: ", infotext['ssw'])
        # print("[DEBUG] ST: ", infotext['st'])
        print("[DEBUG] QT: ", infotext['qt'])
        # print(f"Step: {step}")
        # print(f"Infected left: {infotext['it']}")
        # if doSingletonReduction:
        #     print(f"> Non-singletons infected: {infotext['it']-infotext['is']}")
        #     print(f"> Singletons infected: {infotext['is']}")

        # Early Stopping
        if infotext['it']-infotext['is'] == 0 and infotext['is'] == 0 and infotext['qt'] == 0:
            break

        # >>> Infection, recovery and noticeability
        if doSingletonReduction and virusFound:
            print("[INFO] Recovering singletons...")
            temp_infsng_list = list(has_infected_singletons) # Temporary infected singleton list
            for i in temp_infsng_list:
                inf_sng = G.nodes[i]['i_singletons']
                if inf_sng > 0:
                    temp = np.random.binomial(n=inf_sng, p=recovery_rate)

                    G.nodes[i]['i_singletons'] -= temp
                    G.nodes[i]['r_singletons'] += temp
                    
                    infotext['is'] -= temp
                    infotext['it'] -= temp
                    infotext['rs'] += temp
                    infotext['rt'] += temp
                    if doVisualization and temp > 0:
                        nodes_to_draw.add(i)
                else:
                    has_infected_singletons.remove(i)

        if doSingletonReduction:
            print("[INFO] Infecting singletons and neighbors + Recovering infected nodes...")
        else:
            print("[INFO] Infecting neighbors + Recovering infected nodes...")
            
        temp_inf_list = list(infected_list)
        for i in temp_inf_list:
            # Quarantine current infected node
            if np.random.sample() < noticeability_rate and not i in quarantined_list and not quarantine_length == 0:
                quarantined_origin_list[i] = quarantine_length
                quarantined_list.add(i)
                virusFound = True

            # Infect adjacent nodes
            isQuarantined = i in quarantined_list
            if not isQuarantined:
                adj = G.neighbors(i)
                for j in adj:
                    
                    #Decide infection rate for node j
                    if not infection_islist:
                        new_infection_rate = infection_rate
                    elif G.out_degree(j) > average_out_degree and infection_islist:
                        new_infection_rate = infection_rate[0]
                    elif G.out_degree(j) <= average_out_degree and infection_islist:
                        new_infection_rate = infection_rate[1]
                    

                    if G.nodes[j].get('state', 0) == 0 and np.random.sample() < new_infection_rate:
                        infected_list.add(j)
                        G.nodes[j]['state'] = 1

                        infotext['st'] -= 1
                        infotext['it'] += 1
                        
                        if doVisualization:
                            nodes_to_draw.add(j)

            # Infect singletons
            if doSingletonReduction and not isQuarantined:
                temp = np.random.binomial(n=G.nodes[i]['s_singletons'], p=new_infection_rate)
                if temp > 0:
                    has_infected_singletons.add(i)
                    G.nodes[i]['s_singletons'] -= temp
                    G.nodes[i]['i_singletons'] += temp

                    infotext['st'] -= temp
                    infotext['it'] += temp
                    infotext['ss'] -= temp
                    infotext['ssw'] -= temp
                    infotext['is'] += temp
                    
                    if doVisualization:
                        nodes_to_draw.add(i)

            # Recover infected nodes
            if np.random.sample() < recovery_rate and virusFound:
                infected_list.remove(i)
                G.nodes[i]['state'] = 2

                infotext['it'] -= 1
                infotext['rt'] += 1
                
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

            infotext_total.append(dict(infotext))

    if doRenderInfoText:
        return nodelist_total, nodecolors_total, edgecolors_total, options, infotext_total, constants_total
    return nodelist_total, nodecolors_total, edgecolors_total, options

def work(i, G, pos, nodelist_total, nodecolors_total, edgecolors_total, options, infotext_total: list = None, constants_total: dict = None, doDrawEdges: bool = False):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    
    print(f"Starting graph image: {i}")
    fig, ax = plt.subplots()
    
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

    if doDrawEdges:
        nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.25)

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
            
    ax.axis("off")

    if not infotext_total is None:
        info = infotext_total[i]
        text_to_render = ""
        text_to_render += f"\nStep: {i+1}" 
        text_to_render += f"\nTotal Nodes: {constants_total['tn']}"
        text_to_render += f"\nTotal Susceptible: {info['st']}"
        text_to_render += f"\nTotal Infected: {info['it']}"
        text_to_render += f"\nTotal Recovered: {info['rt']}"
        text_to_render += f"\nTotal Isolated: {constants_total['xt']}"
        text_to_render += "\n" + "~"*10
        text_to_render += f"\nTotal Quarantined: {info['qt']}"
        text_to_render += f"\nSusceptible Singletons: {info['ssw']}" # or ['ss']
        text_to_render += f"\nInfected Singletons: {info['is']}"
        text_to_render += f"\nRecovered Singletons: {info['rs']}"
    plt.text(-0.25, 0.95, s=text_to_render, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    fig.subplots_adjust(left=0.2) 
    fig.tight_layout()
    fig.savefig(f"graphs/graph_{i}.png", format="PNG", bbox_inches='tight')
    print(f"Saved graph_{i}.png!")
    plt.close(fig)
