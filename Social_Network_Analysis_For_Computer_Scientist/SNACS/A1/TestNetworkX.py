import networkx as nx
import matplotlib.pyplot as plt
from networkx import degree
from networkx.algorithms.threshold import degree_sequence

#Step1: demo
# G = nx.Graph()
#
# G.add_nodes_from([1,2,3,4])
#
# G.add_edges_from([(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)])
#
# nx.draw(G, with_labels=True, node_color='lightblue', node_size=128, font_size=12)
# plt.show()
#
# print("number of nodes:", G.number_of_nodes())
# print("number of edges:", G.number_of_edges())
# print("dict of degree:", dict(G.degree()))
# print("connection", nx.is_connected(G))
# print("shortest path:", nx.shortest_path(G, source=1, target=4))

#Step2: read&write from csv format
#Step3: degree distribution
G = nx.read_edgelist("small-gephiready.tsv", delimiter="\t", create_using=nx.Graph(), nodetype=str, data=False, comments=None)

degree_sequence = [d for n, d in G.degree()]

plt.hist(degree_sequence)
plt.title("degree_distribution")
plt.xlabel("degree")
plt.ylabel("count")
plt.show()
print("number of nodes:", G.number_of_nodes())
print("number of edges:", G.number_of_edges())

#nx.write_edgelist(G, "small-gephiready.txt")