import networkx as nx
G = nx.read_gml('./Internet Zoo/Nsfnet.gml')
G.edges
print(G.edges)