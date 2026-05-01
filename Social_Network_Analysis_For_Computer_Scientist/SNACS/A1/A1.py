# Listing 1: Load graph and count edges
import networkx as nx
import matplotlib.pyplot as plt

# Read edge list (tab-separated, directed)
G = nx.read_edgelist("snacs2025-student4213211-medium.tsv", delimiter="\t", create_using=nx.DiGraph(), nodetype=int)

print("Number of directed edges:", G.number_of_edges())
print("Number of nodes:", G.number_of_nodes())

# Listing 2: Compute and plot in-/out-degree distributions

# Extract degree sequences
indeg = [d for _, d in G.in_degree()]
outdeg = [d for _, d in G.out_degree()]

# Plot histogram (log-log scale)
plt.figure()
plt.hist(indeg, bins=50, log=True)
plt.xlabel("In-degree")
plt.ylabel("Frequency")
plt.title("In-degree distribution (medium.tsv)")
plt.show()

plt.figure()
plt.hist(outdeg, bins=50, log=True)
plt.xlabel("Out-degree")
plt.ylabel("Frequency")
plt.title("Out-degree distribution (medium.tsv)")
plt.show()
