# snacs_q2.py
import argparse
import collections
import time
import networkx as nx
import matplotlib.pyplot as plt


def read_graph(tsv_path: str) -> nx.DiGraph:
    """Read a tab-separated edge list as a directed graph of ints."""
    G = nx.read_edgelist(
        tsv_path,
        delimiter="\t",
        create_using=nx.DiGraph(),
        nodetype=int,
        data=False,
    )
    return G


def q21_q22_counts(G: nx.DiGraph):
    return G.number_of_edges(), G.number_of_nodes()


def degree_sequences(G: nx.DiGraph):
    indeg = [d for _, d in G.in_degree()]
    outdeg = [d for _, d in G.out_degree()]
    return indeg, outdeg


def plot_hist(data, title, xlabel, out_png, bins=50, log=True):
    plt.figure()
    plt.hist(data, bins=bins, log=log)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def q23_plots(G: nx.DiGraph, prefix: str):
    indeg, outdeg = degree_sequences(G)
    plot_hist(indeg, f"In-degree distribution ({prefix})", "In-degree",
              f"{prefix}_in_degree.png")
    plot_hist(outdeg, f"Out-degree distribution ({prefix})", "Out-degree",
              f"{prefix}_out_degree.png")


def largest_wcc(G: nx.DiGraph):
    L = max(nx.weakly_connected_components(G), key=len)
    H = G.subgraph(L).copy()
    return H


def largest_scc(G: nx.DiGraph):
    L = max(nx.strongly_connected_components(G), key=len)
    H = G.subgraph(L).copy()
    return H


def q24_components(G: nx.DiGraph):
    num_wcc = nx.number_weakly_connected_components(G)
    num_scc = nx.number_strongly_connected_components(G)
    H_wcc = largest_wcc(G)
    H_scc = largest_scc(G)
    return {
        "num_wcc": num_wcc,
        "num_scc": num_scc,
        "largest_wcc_nodes": H_wcc.number_of_nodes(),
        "largest_wcc_edges": H_wcc.number_of_edges(),
        "largest_scc_nodes": H_scc.number_of_nodes(),
        "largest_scc_edges": H_scc.number_of_edges(),
    }


def q25_clustering(G: nx.DiGraph):
    """Directed & undirected average clustering. Undirected uses simple projection."""
    # Directed GC (Fagiolo-style in NX) – be aware of definition differences.
    try:
        c_dir = nx.average_clustering(G)  # on DiGraph: Fagiolo definition
    except Exception:
        c_dir = float("nan")

    # Undirected projection
    c_undir = nx.average_clustering(G.to_undirected())

    # Triangles (undirected) for reporting (optional)
    tri_undir = sum(nx.triangles(G.to_undirected()).values()) // 3
    return {"clustering_directed": c_dir, "clustering_undirected": c_undir, "triangles_undirected": tri_undir}


def all_pairs_shortest_path_lengths_undirected(H: nx.Graph):
    """Return a Counter of distances for an undirected connected graph H (can be large)."""
    dist_hist = collections.Counter()
    nodes = list(H.nodes())
    for i, s in enumerate(nodes):
        lengths = nx.single_source_shortest_path_length(H, s)
        for t, d in lengths.items():
            # count unordered pairs once
            if t > s:
                dist_hist[d] += 1
    return dist_hist


def q26_distance_distribution(G: nx.DiGraph, prefix: str, sample: int = 0):
    """Compute distance distribution on the largest WCC (undirected),
    optionally by sampling 'sample' sources to speed up on large graphs."""
    H = largest_wcc(G).to_undirected()

    if sample and sample < H.number_of_nodes():
        import random
        sources = random.sample(list(H.nodes()), sample)
        dist_hist = collections.Counter()
        for s in sources:
            lengths = nx.single_source_shortest_path_length(H, s)
            for t, d in lengths.items():
                if t != s:
                    dist_hist[d] += 1
        # this is not normalized to unordered pairs; it’s a sampled view (fine for plots)
    else:
        dist_hist = all_pairs_shortest_path_lengths_undirected(H)

    # Plot
    xs = sorted(dist_hist.keys())
    ys = [dist_hist[x] for x in xs]
    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel("Shortest path length")
    plt.ylabel("Count of node pairs")
    plt.title(f"Distance distribution on largest WCC ({prefix})")
    plt.tight_layout()
    out_png = f"{prefix}_distance_distribution.png"
    plt.savefig(out_png)
    plt.close()
    return {"distance_hist": dist_hist, "plot": out_png}


def q27_average_path_length(G: nx.DiGraph, sample: int = 0):
    """Average shortest path length on largest WCC (undirected).
    Exact if sample==0; else estimate by sampling sources."""
    H = largest_wcc(G).to_undirected()
    if sample and sample < H.number_of_nodes():
        import random
        total = 0
        count = 0
        for s in random.sample(list(H.nodes()), sample):
            lengths = nx.single_source_shortest_path_length(H, s)
            for t, d in lengths.items():
                if t != s:
                    total += d
                    count += 1
        return total / count
    else:
        return nx.average_shortest_path_length(H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="Path to edge list (TSV: src<TAB>dst)")
    ap.add_argument("--prefix", default="medium", help="Prefix for output files")
    ap.add_argument("--sample", type=int, default=0,
                    help="Sampling sources for Q2.6/Q2.7 (0 = exact; >0 = sampled)")
    args = ap.parse_args()

    t0 = time.time()
    G = read_graph(args.tsv)
    assert G.is_directed()

    # Q2.1 & Q2.2
    m, n = q21_q22_counts(G)
    print(f"Q2.1 #Edges = {m}")
    print(f"Q2.2 #Nodes = {n}")

    # Q2.3
    q23_plots(G, args.prefix)
    print(f"Q2.3 Plots saved: {args.prefix}_in_degree.pdf, {args.prefix}_out_degree.pdf")

    # Q2.4
    comp = q24_components(G)
    print("Q2.4",
          f"#WCC={comp['num_wcc']}, #SCC={comp['num_scc']}, "
          f"Largest WCC=({comp['largest_wcc_nodes']} nodes, {comp['largest_wcc_edges']} edges), "
          f"Largest SCC=({comp['largest_scc_nodes']} nodes, {comp['largest_scc_edges']} edges)")

    # Q2.5
    clus = q25_clustering(G)
    print(f"Q2.5 Clustering (directed)={clus['clustering_directed']:.6f}, "
          f"(undirected)={clus['clustering_undirected']:.6f}, "
          f"triangles_undirected={clus['triangles_undirected']}")

    # Q2.6
    dist = q26_distance_distribution(G, args.prefix, sample=args.sample)
    print(f"Q2.6 Distance distribution plot: {dist['plot']} "
          f"(unique distances: {len(dist['distance_hist'])})")

    # Q2.7
    apl = q27_average_path_length(G, sample=args.sample)
    print(f"Q2.7 Average path length (largest WCC, undirected) = {apl:.6f}")

    print(f"Done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
