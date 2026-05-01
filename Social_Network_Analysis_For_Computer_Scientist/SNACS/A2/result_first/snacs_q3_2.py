# snacs_q3_for_v10_1.py — SNACS Assignment 2 Q3 (适配 NetworKit 10.1)
# --------------------------------------------------------------
# 用法:
#   python snacs_q3_for_v10_1.py --tsv data/twitter-small.tsv --prefix small --sample 500
#   python snacs_q3_for_v10_1.py --tsv data/twitter-larger.tsv --prefix larger --sample 1000
# --------------------------------------------------------------

import argparse
import csv
import re
import time
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import networkit as nk
from scipy.stats import kendalltau


# ----------------------------------------------------------------------
# Q3.1  Parse raw tweets → weighted mention graph
# ----------------------------------------------------------------------
def extract_mention_graph(in_tsv: str):
    """从原始 Twitter 文件中解析 mention 图并构建 NetworKit 图"""
    # 匹配 @username 格式，用户名可以包含字母、数字和下划线
    mention_re = re.compile(r'@([A-Za-z0-9_]+)')
    idmap = {}
    next_id = 0
    edges = collections.Counter()
    skipped = 0

    print(f"[INFO] Parsing tweets from {in_tsv}...")
    with open(in_tsv, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            # 每条推文应有3个字段：timestamp, user, content
            if len(parts) < 3:
                skipped += 1
                continue
            timestamp, user, text = parts[0], parts[1], parts[2]

            # 跳过空用户名
            if not user or not user.strip():
                skipped += 1
                continue

            user = user.strip()

            # 为发推用户分配 ID
            if user not in idmap:
                idmap[user] = next_id
                next_id += 1
            u = idmap[user]

            # 查找所有被提及的用户
            for mentioned in mention_re.findall(text):
                # 忽略自我提及
                if mentioned == user:
                    continue
                if mentioned not in idmap:
                    idmap[mentioned] = next_id
                    next_id += 1
                v = idmap[mentioned]
                edges[(u, v)] += 1

    n = len(idmap)
    m = len(edges)
    print(f"[INFO] Parsed {n} users, {m} mention links (skipped {skipped} lines)")

    # 构建有向加权图
    G = nk.Graph(n, weighted=True, directed=True)
    for (u, v), w in edges.items():
        G.addEdge(u, v, w)

    return G, idmap


def write_weighted_edges(G: nk.Graph, idmap: dict, out_csv: str):
    """将加权边列表写入 CSV 文件"""
    invmap = {v: k for k, v in idmap.items()}
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "weight"])
        for u, v in G.iterEdges():
            weight = G.weight(u, v)
            w.writerow([invmap.get(u, u), invmap.get(v, v), int(weight)])
    print(f"[INFO] Weighted edge list written to {out_csv}")


# ----------------------------------------------------------------------
# Q3.2 Basic statistics & clustering (v10.1)
# ----------------------------------------------------------------------
def basic_stats(G: nk.Graph):
    """计算基本网络统计信息"""
    n = G.numberOfNodes()
    m = G.numberOfEdges()
    density = nk.graphtools.density(G)

    # 弱连通分量
    wc = nk.components.WeaklyConnectedComponents(G)
    wc.run()
    num_wcc = wc.numberOfComponents()
    wcc_sizes = wc.getComponentSizes()
    largest_wcc = max(wcc_sizes.values()) if wcc_sizes else 0

    # 强连通分量
    sc = nk.components.StronglyConnectedComponents(G)
    sc.run()
    num_scc = sc.numberOfComponents()
    scc_sizes = sc.getComponentSizes()
    largest_scc = max(scc_sizes.values()) if scc_sizes else 0

    return {
        "n": n, "m": m, "density": density,
        "num_wcc": num_wcc, "num_scc": num_scc,
        "largest_wcc": largest_wcc, "largest_scc": largest_scc
    }


def clustering_stats(G: nk.Graph):
    """NetworKit 10.1 聚类统计接口"""
    # 转换为无向图来计算聚类系数
    U = nk.graphtools.toUndirected(G)

    # 局部聚类系数
    lcc = nk.centrality.LocalClusteringCoefficient(U)
    lcc.run()
    lcc_scores = lcc.scores()
    avg_clustering = np.mean(lcc_scores)

    # 三角形计数
    # 使用公式: 对于每个节点 v，三角形数 = c(v) * deg(v) * (deg(v) - 1) / 2
    # 总三角形数 = sum / 3（每个三角形被三个顶点各计数一次）
    triangle_count = 0.0
    for v in U.iterNodes():
        deg = U.degree(v)
        if deg > 1:
            c_v = lcc_scores[v]
            # c_v = 实际三角形边数 / 可能的三角形边数
            # 实际三角形边数 = c_v * deg * (deg - 1) / 2
            triangle_count += c_v * deg * (deg - 1) / 2.0

    # 每个三角形被计数3次（每个顶点一次）
    triangles = int(round(triangle_count / 3.0))

    return {"clustering_avg": avg_clustering, "triangles": triangles}


def plot_degree_distributions(G: nk.Graph, prefix: str):
    """绘制入度和出度分布"""
    print("[INFO] Plotting degree distributions...")

    indegrees = [G.degreeIn(v) for v in G.iterNodes()]
    outdegrees = [G.degreeOut(v) for v in G.iterNodes()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 入度分布
    ax1.hist(indegrees, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel("In-degree")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"In-degree Distribution ({prefix})")
    ax1.set_yscale('log')

    # 出度分布
    ax2.hist(outdegrees, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel("Out-degree")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Out-degree Distribution ({prefix})")
    ax2.set_yscale('log')

    plt.tight_layout()
    out_png = f"{prefix}_degree_distributions.png"
    plt.savefig(out_png, dpi=150)
    plt.close()

    return out_png


def get_giant_component(G: nk.Graph):
    """提取最大弱连通分量（巨连通分量）"""
    wc = nk.components.WeaklyConnectedComponents(G)
    wc.run()

    # 找到最大连通分量的 ID
    component_sizes = wc.getComponentSizes()
    largest_comp_id = max(component_sizes.keys(), key=lambda k: component_sizes[k])

    # 提取该连通分量的节点
    nodes_in_giant = [v for v in G.iterNodes() if wc.componentOfNode(v) == largest_comp_id]

    # 创建子图
    G_giant = nk.graphtools.subgraphFromNodes(G, nodes_in_giant)

    print(f"[INFO] Giant component: {G_giant.numberOfNodes()} nodes, {G_giant.numberOfEdges()} edges")
    return G_giant


def distance_distribution(G: nk.Graph, prefix: str, sample: int = 500):
    """计算并绘制距离分布（在无向巨连通分量上）"""
    print("[INFO] Computing distance distribution on giant component...")

    # 转换为无向图
    U = nk.graphtools.toUndirected(G)

    # 采样一些节点计算距离
    nodes = list(U.iterNodes())
    sample_size = min(sample, len(nodes))

    if sample_size >= len(nodes):
        sampled_nodes = nodes
    else:
        sampled_nodes = random.sample(nodes, sample_size)

    print(f"  Sampling {len(sampled_nodes)} nodes for distance computation...")

    all_distances = []

    for i, source in enumerate(sampled_nodes):
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i + 1}/{len(sampled_nodes)} nodes")

        try:
            bfs = nk.distance.BFS(U, source, storePaths=False)
            bfs.run()
            dists = bfs.getDistances()

            # 收集有效距离 - 使用 np.isfinite 来过滤
            for d in dists:
                if d > 0 and np.isfinite(d):  # 排除 0, inf, nan
                    all_distances.append(int(d))

        except Exception as e:
            print(f"    Warning: Failed to compute distances from node {source}: {e}")
            continue

    if not all_distances:
        print("[WARNING] No valid distances computed")
        return None, None

    # 统计
    avg_distance = float(np.mean(all_distances))
    max_dist = max(all_distances)
    min_dist = min(all_distances)

    print(f"  Collected {len(all_distances)} distance measurements")
    print(f"  Distance stats: min={min_dist}, max={max_dist}, avg={avg_distance:.4f}")

    # 绘图
    plt.figure(figsize=(8, 6))

    # 智能选择 bins
    dist_range = max_dist - min_dist + 1
    if dist_range <= 50:
        bins = list(range(min_dist, max_dist + 2))
    else:
        bins = 50

    plt.hist(all_distances, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel("Distance (hops)")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"Distance Distribution - Giant Component ({prefix})")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png = f"{prefix}_distance_distribution.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Plot saved: {out_png}")

    return avg_distance, out_png


def approx_diameter_and_distance(G: nk.Graph, sample: int = 500):
    """近似计算直径和平均距离"""
    print("[INFO] Approximating diameter and average distance...")

    U = nk.graphtools.toUndirected(G)

    # 近似直径
    diam_algo = nk.distance.ApproxDiameter(U, algo=1, nSamples=sample)
    diam_algo.run()
    approx_diam = diam_algo.getDiameter()

    # 近似平均距离
    avg_dist_algo = nk.distance.ApproxAvgDistance(U, nSamples=sample)
    avg_dist_algo.run()
    avg_dist = avg_dist_algo.getAvgDistance()

    return approx_diam, avg_dist


# ----------------------------------------------------------------------
# Q3.3 Centralities
# ----------------------------------------------------------------------
def compute_centralities(G: nk.Graph, topk=20):
    """
    计算三种中心性度量。
    对于有向图：
    - 度中心性：使用总度数（入度+出度）
    - 接近中心性和介数中心性：基于有向路径
    """
    print("[INFO] Computing centralities (approximate)...")

    # 度中心性：使用总度数
    deg_scores = [G.degreeIn(v) + G.degreeOut(v) for v in G.iterNodes()]

    # 接近中心性（有向）
    close = nk.centrality.ApproxCloseness(G, nSamples=min(500, G.numberOfNodes()), normalized=False)
    close.run()
    close_scores = close.scores()

    # 介数中心性（有向，近似）
    betw = nk.centrality.ApproxBetweenness(G, nSamples=min(200, G.numberOfNodes()))
    betw.run()
    betw_scores = betw.scores()

    ids = list(range(G.numberOfNodes()))

    def topk_sorted(scores):
        return sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)[:topk]

    return {
        "degree": topk_sorted(deg_scores),
        "closeness": topk_sorted(close_scores),
        "betweenness": topk_sorted(betw_scores)
    }


def kendall_tau_similarity(rank_a, rank_b):
    """计算两个排名之间的 Kendall Tau 相关系数"""
    users_a = [u for u, _ in rank_a]
    users_b = [u for u, _ in rank_b]
    common = list(set(users_a) & set(users_b))

    if len(common) < 3:
        return None

    # 在各自排名中的位置
    idx_a = [users_a.index(u) for u in common]
    idx_b = [users_b.index(u) for u in common]

    tau, pval = kendalltau(idx_a, idx_b)
    return tau


# ----------------------------------------------------------------------
# Q3.4 Community detection
# ----------------------------------------------------------------------
def detect_communities(G: nk.Graph, prefix: str):
    """在图上运行 Louvain 社区检测"""
    print("[INFO] Running Louvain community detection...")

    # Louvain 算法在无向图上效果更好
    U = nk.graphtools.toUndirected(G)

    # NetworKit 10.1 的 PLM (Parallel Louvain Method)
    louvain = nk.community.PLM(U, refine=True)
    louvain.run()
    partition = louvain.getPartition()

    num_comms = partition.numberOfSubsets()
    sizes = [partition.subsetSizeMap()[i] for i in range(num_comms)]

    # 绘制社区大小分布
    plt.figure(figsize=(8, 6))
    plt.hist(sizes, bins=min(50, num_comms), edgecolor='black', alpha=0.7)
    plt.xlabel("Community size")
    plt.ylabel("Frequency")
    plt.title(f"Community Size Distribution ({prefix})")
    plt.yscale('log')
    plt.tight_layout()

    out_png = f"{prefix}_community_sizes.png"
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"[INFO] Found {num_comms} communities")
    print(f"[INFO] Top 5 largest communities: {sorted(sizes, reverse=True)[:5]}")

    return {"num_communities": num_comms, "plot": out_png, "sizes": sizes}


# ----------------------------------------------------------------------
# Q3.5 Weight distribution
# ----------------------------------------------------------------------
def plot_weight_distribution(G: nk.Graph, prefix: str):
    """绘制边权重分布"""
    print("[INFO] Plotting weight distribution...")

    weights = [G.weight(u, v) for u, v in G.iterEdges()]

    plt.figure(figsize=(8, 6))
    plt.hist(weights, bins=min(100, len(set(weights))), edgecolor='black', alpha=0.7)
    plt.xlabel("Edge weight (number of mentions)")
    plt.ylabel("Frequency")
    plt.title(f"Edge Weight Distribution ({prefix})")
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()

    out_png = f"{prefix}_weight_distribution.png"
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"[INFO] Weight stats: min={min(weights)}, max={max(weights)}, mean={np.mean(weights):.2f}")

    return out_png


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SNACS Assignment 2 Q3: Twitter Network Analysis")
    ap.add_argument("--tsv", required=True, help="Input TSV file (twitter-small.tsv / twitter-larger.tsv)")
    ap.add_argument("--prefix", default="small", help="Output prefix for files and plots")
    ap.add_argument("--sample", type=int, default=500, help="Sampling size for approximate algorithms")
    args = ap.parse_args()

    print("=" * 70)
    print(f"SNACS Assignment 2 Q3: Twitter Mention Network Analysis")
    print(f"Dataset: {args.tsv}")
    print(f"Prefix: {args.prefix}")
    print("=" * 70)

    t0 = time.time()

    # Q3.1: 解析 mention 图
    print("\n--- Q3.1: Extracting mention graph ---")
    G, idmap = extract_mention_graph(args.tsv)
    write_weighted_edges(G, idmap, f"{args.prefix}_weighted_edges.csv")

    # Q3.2: 基本统计
    print("\n--- Q3.2: Computing basic statistics ---")
    stats = basic_stats(G)
    print(f"\nBasic Statistics:")
    print(f"  Nodes: {stats['n']}")
    print(f"  Edges: {stats['m']}")
    print(f"  Density: {stats['density']:.6f}")
    print(f"  Weakly connected components: {stats['num_wcc']} (largest: {stats['largest_wcc']})")
    print(f"  Strongly connected components: {stats['num_scc']} (largest: {stats['largest_scc']})")

    clus = clustering_stats(G)
    print(f"  Average clustering coefficient: {clus['clustering_avg']:.6f}")
    print(f"  Number of triangles: {clus['triangles']}")

    # 度分布图
    deg_plot = plot_degree_distributions(G, args.prefix)
    print(f"  Degree distribution plots saved: {deg_plot}")

    # 提取巨连通分量并计算距离
    G_giant = get_giant_component(G)
    avg_dist_sampled, dist_plot = distance_distribution(G_giant, args.prefix, sample=args.sample)
    if avg_dist_sampled:
        print(f"  Average distance (sampled): {avg_dist_sampled:.4f}")
        print(f"  Distance distribution plot: {dist_plot}")

    approx_diam, approx_avg_dist = approx_diameter_and_distance(G_giant, sample=args.sample)
    print(f"  Approximate diameter: {approx_diam}")
    print(f"  Approximate average distance: {approx_avg_dist:.4f}")

    # Q3.3: 中心性
    print("\n--- Q3.3: Computing centralities ---")
    centr = compute_centralities(G, topk=20)

    print("\nTop 20 by Degree Centrality:")
    for i, (node, score) in enumerate(centr["degree"][:5], 1):
        print(f"  {i}. Node {node}: {score}")
    print("  ...")

    print("\nTop 20 by Closeness Centrality:")
    for i, (node, score) in enumerate(centr["closeness"][:5], 1):
        print(f"  {i}. Node {node}: {score:.6f}")
    print("  ...")

    print("\nTop 20 by Betweenness Centrality:")
    for i, (node, score) in enumerate(centr["betweenness"][:5], 1):
        print(f"  {i}. Node {node}: {score:.6f}")
    print("  ...")

    tau_dc = kendall_tau_similarity(centr["degree"], centr["closeness"])
    tau_db = kendall_tau_similarity(centr["degree"], centr["betweenness"])
    tau_cb = kendall_tau_similarity(centr["closeness"], centr["betweenness"])

    print(f"\nRanking Similarity (Kendall Tau):")
    print(f"  Degree vs Closeness: {tau_dc:.4f}" if tau_dc else "  Degree vs Closeness: N/A")
    print(f"  Degree vs Betweenness: {tau_db:.4f}" if tau_db else "  Degree vs Betweenness: N/A")
    print(f"  Closeness vs Betweenness: {tau_cb:.4f}" if tau_cb else "  Closeness vs Betweenness: N/A")

    # Q3.4: 社区检测（在巨连通分量上）
    print("\n--- Q3.4: Community detection ---")
    comm = detect_communities(G_giant, args.prefix)
    print(f"  Community plot saved: {comm['plot']}")

    # Q3.5: 权重分布
    print("\n--- Q3.5: Weight distribution ---")
    wplot = plot_weight_distribution(G, args.prefix)
    print(f"  Weight distribution plot: {wplot}")

    print("\n" + "=" * 70)
    print(f"Total time: {time.time() - t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()