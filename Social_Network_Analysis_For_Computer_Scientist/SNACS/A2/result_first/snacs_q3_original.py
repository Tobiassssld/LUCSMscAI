# snacs_q3_original.py — Assignment 2 Q3 (Optimized with Louvain)
# Author: Tobias Liu
# Compatible with: Python ≥3.9, NetworkX ≥3.0
# Fast version: community detection via Louvain (if installed)
# Usage:
#   python snacs_q3_original.py --tsv data/twitter-larger.tsv --prefix larger --sample 1000

import argparse
import collections
import csv
import re
import time
import random
import networkx as nx
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Q3.1  ── Parse raw tweets  →  weighted mention graph
# ----------------------------------------------------------------------
def extract_mention_graph(in_tsv: str) -> nx.DiGraph:
    mention_re = re.compile(r'@([A-Za-z0-9_]+)')
    G = nx.DiGraph()
    skipped = 0

    with open(in_tsv, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                skipped += 1
                continue
            _, user, text = parts[0], parts[1], parts[2]
            if not user:
                skipped += 1
                continue
            for mentioned in mention_re.findall(text):
                if mentioned == user:
                    continue
                if G.has_edge(user, mentioned):
                    G[user][mentioned]['weight'] += 1
                else:
                    G.add_edge(user, mentioned, weight=1)
    print(f"[INFO] Parsed {G.number_of_nodes()} users, {G.number_of_edges()} mention links (skipped {skipped} lines)")
    return G


def write_weighted_edges(G: nx.DiGraph, out_csv: str):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "weight"])
        for u, v, d in G.edges(data=True):
            w.writerow([u, v, d.get("weight", 1)])
    print(f"[INFO] Weighted edge list written to {out_csv}")


# ----------------------------------------------------------------------
# Basic statistics (Q3.2)
# ----------------------------------------------------------------------
def q31_q32_counts(G):
    return G.number_of_edges(), G.number_of_nodes()


def q34_components(G: nx.DiGraph):
    num_wcc = nx.number_weakly_connected_components(G)
    num_scc = nx.number_strongly_connected_components(G)
    H_wcc = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
    H_scc = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
    return {
        "num_wcc": num_wcc,
        "num_scc": num_scc,
        "largest_wcc_nodes": H_wcc.number_of_nodes(),
        "largest_wcc_edges": H_wcc.number_of_edges(),
        "largest_scc_nodes": H_scc.number_of_nodes(),
        "largest_scc_edges": H_scc.number_of_edges(),
    }


def q35_clustering(G: nx.DiGraph):
    try:
        c_dir = nx.average_clustering(G)
    except Exception:
        c_dir = float("nan")
    c_undir = nx.average_clustering(G.to_undirected())
    tri_undir = sum(nx.triangles(G.to_undirected()).values()) // 3
    return dict(
        clustering_directed=c_dir,
        clustering_undirected=c_undir,
        triangles_undirected=tri_undir,
    )


# ----------------------------------------------------------------------
# Q3.3 ── Centrality measures
# ----------------------------------------------------------------------
def compute_centralities(G: nx.DiGraph, topk: int = 20):
    print("[INFO] Computing centralities...")
    degree_c = nx.degree_centrality(G)
    closeness_c = nx.closeness_centrality(G)
    betweenness_c = nx.betweenness_centrality(G, k=min(50, G.number_of_nodes()))

    def top(d):
        return sorted(d.items(), key=lambda x: x[1], reverse=True)[:topk]

    return {
        "degree": top(degree_c),
        "closeness": top(closeness_c),
        "betweenness": top(betweenness_c),
    }


def kendall_tau_similarity(rank_a, rank_b):
    from scipy.stats import kendalltau
    users_a = [u for u, _ in rank_a]
    users_b = [u for u, _ in rank_b]
    common = list(set(users_a) & set(users_b))
    if len(common) < 3:
        return None
    idx_a = [users_a.index(u) for u in common]
    idx_b = [users_b.index(u) for u in common]
    tau, _ = kendalltau(idx_a, idx_b)
    return tau


# ----------------------------------------------------------------------
# Q3.4 ── Fast Community detection
# ----------------------------------------------------------------------
def detect_communities(G, prefix: str):
    """Try Louvain (python-louvain); fallback to NetworkX greedy modularity."""
    U = G.to_undirected()
    try:
        import community as community_louvain
        print("[INFO] Running Louvain community detection (fast C implementation)...")
        partition = community_louvain.best_partition(U, random_state=42)
        num_comms = len(set(partition.values()))
        print(f"[INFO] Detected {num_comms} communities (Louvain).")
        sizes = collections.Counter(partition.values())
    except Exception:
        print("[WARN] Louvain not installed; falling back to slow NetworkX greedy modularity.")
        comms = list(nx.community.greedy_modularity_communities(U))
        num_comms = len(comms)
        sizes = {i: len(c) for i, c in enumerate(comms)}

    plt.figure()
    plt.hist(sizes.values(), bins=50)
    plt.xlabel("Community size")
    plt.ylabel("Frequency")
    plt.title(f"Community size distribution ({prefix})")
    plt.tight_layout()
    out_png = f"{prefix}_community_sizes.png"
    plt.savefig(out_png)
    plt.close()
    return {"num_communities": num_comms, "plot": out_png}


# ----------------------------------------------------------------------
# Q3.5 ── Weight distribution
# ----------------------------------------------------------------------
def plot_weight_distribution(G: nx.DiGraph, prefix: str):
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    plt.figure()
    plt.hist(weights, bins=100, log=True)
    plt.xlabel("Edge weight (#mentions)")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"Weight distribution ({prefix})")
    plt.tight_layout()
    out_png = f"{prefix}_weight_distribution.png"
    plt.savefig(out_png)
    plt.close()
    return out_png


# ----------------------------------------------------------------------
# Q3.6 ── Distance distribution & avg. path
# ----------------------------------------------------------------------
def q36_distance_distribution(G: nx.DiGraph, prefix: str, sample: int = 0):
    H = G.subgraph(max(nx.weakly_connected_components(G), key=len)).to_undirected()
    nodes = list(H.nodes())
    dist_hist = collections.Counter()

    if sample and sample < len(nodes):
        for s in random.sample(nodes, sample):
            lengths = nx.single_source_shortest_path_length(H, s)
            for t, d in lengths.items():
                if t != s:
                    dist_hist[d] += 1
    else:
        for s in nodes:
            lengths = nx.single_source_shortest_path_length(H, s)
            for t, d in lengths.items():
                if t > s:
                    dist_hist[d] += 1

    xs = sorted(dist_hist)
    ys = [dist_hist[x] for x in xs]
    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel("Shortest path length")
    plt.ylabel("Count of node pairs")
    plt.title(f"Distance distribution ({prefix})")
    plt.tight_layout()
    out_png = f"{prefix}_distance_distribution.png"
    plt.savefig(out_png)
    plt.close()
    return {"hist": dist_hist, "plot": out_png}


def q37_average_path_length(G: nx.DiGraph, sample: int = 0):
    H = G.subgraph(max(nx.weakly_connected_components(G), key=len)).to_undirected()
    if sample and sample < H.number_of_nodes():
        total = count = 0
        for s in random.sample(list(H.nodes()), sample):
            lengths = nx.single_source_shortest_path_length(H, s)
            for t, d in lengths.items():
                if t != s:
                    total += d
                    count += 1
        return total / count
    else:
        return nx.average_shortest_path_length(H)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="Input TSV file (e.g., twitter-larger.tsv)")
    ap.add_argument("--prefix", default="small", help="Output prefix")
    ap.add_argument("--sample", type=int, default=0, help="Sampling for distance calc")
    ap.add_argument("--parse_only", action="store_true", help="Only parse and save weighted edge list")
    args = ap.parse_args()

    t0 = time.time()
    G = extract_mention_graph(args.tsv)
    write_weighted_edges(G, f"{args.prefix}_weighted_edges.csv")
    if args.parse_only:
        return

    edges, nodes = q31_q32_counts(G)
    comp = q34_components(G)
    clus = q35_clustering(G)
    density = nx.density(G)
    print(f"[INFO] N={nodes}, M={edges}, density={density:.6f}")
    print(f"[INFO] Components: {comp}")
    print(f"[INFO] Clustering: {clus}")

    centr = compute_centralities(G)
    tau_deg_close = kendall_tau_similarity(centr["degree"], centr["closeness"])
    tau_deg_betw = kendall_tau_similarity(centr["degree"], centr["betweenness"])
    print(f"[INFO] Rank similarity (degree–closeness)={tau_deg_close}, (degree–betweenness)={tau_deg_betw}")

    comm = detect_communities(G, args.prefix)
    print(f"[INFO] Community plot saved: {comm['plot']}")

    wplot = plot_weight_distribution(G, args.prefix)
    print(f"[INFO] Weight distribution plot: {wplot}")

    dist = q36_distance_distribution(G, args.prefix, sample=args.sample)
    apl = q37_average_path_length(G, sample=args.sample)
    print(f"[INFO] Distance distribution plot: {dist['plot']}")
    print(f"[INFO] Avg shortest path length (largest WCC) = {apl:.4f}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
