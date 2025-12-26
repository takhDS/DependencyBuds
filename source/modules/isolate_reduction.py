import networkx as nx

def isolate_reduction(G):
    print("Nodes before: ", len(G))
    isolatedNum = list(nx.isolates(G))
    G.remove_nodes_from(isolatedNum)
    isolatedNum = len(isolatedNum)
    print("Nodes after: ", len(G))

    return G, isolatedNum