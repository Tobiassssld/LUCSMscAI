"""
Social Network Analysis - Assignment 2, Exercise 3
Twitter Mention Graph Analysis
"""

import re
import csv
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# 配置数据路径
DATA_DIR = 'data'
OUTPUT_DIR = 'result_2'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== Question 3.1 =====================

def extract_mentions(tweet_content):
    """
    Extract valid Twitter mentions from tweet content.
    Twitter username rules: 1-15 characters, alphanumeric and underscore only
    """
    pattern = r'@([A-Za-z0-9_]{1,15})'
    mentions = re.findall(pattern, tweet_content)
    return [mention.lower() for mention in mentions]


def parse_twitter_data(input_file, output_file):
    """
    Parse Twitter data and extract mention graph.
    Creates weighted edge list from mentions.
    """
    adjacency_dict = defaultdict(lambda: defaultdict(int))
    all_users = set()

    line_count = 0
    error_count = 0

    print(f"Reading from: {input_file}")

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_count += 1
            if line_count % 10000 == 0:
                print(f"Processed {line_count} lines...")

            try:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    error_count += 1
                    continue

                timestamp = parts[0]
                sender = parts[1].lower().strip()
                content = parts[2]

                # Skip empty usernames
                if not sender:
                    error_count += 1
                    continue

                all_users.add(sender)

                # Extract mentions
                mentions = extract_mentions(content)

                for mentioned_user in mentions:
                    if mentioned_user != sender:  # No self-mentions
                        adjacency_dict[sender][mentioned_user] += 1
                        all_users.add(mentioned_user)

            except Exception as e:
                error_count += 1
                continue

    # Write weighted edge list
    print(f"Writing to: {output_file}")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'target', 'weight'])

        for source in adjacency_dict:
            for target, weight in adjacency_dict[source].items():
                writer.writerow([source, target, weight])

    print(f"Parsed {line_count} lines, {error_count} errors")
    print(f"Total users: {len(all_users)}")
    print(f"Total edges: {sum(len(v) for v in adjacency_dict.values())}")

    return output_file


# ===================== Question 3.2 =====================

def load_graph_from_csv(edge_list_file):
    """
    Load directed graph from CSV file with weights.
    """
    G = nx.DiGraph()

    with open(edge_list_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row['source']
            target = row['target']
            weight = int(row['weight'])
            G.add_edge(source, target, weight=weight)

    return G


def compute_network_statistics(edge_list_file):
    """
    Compute comprehensive network statistics.
    """
    print("Loading graph...")
    G = load_graph_from_csv(edge_list_file)

    stats_dict = {}

    # Basic stats
    stats_dict['nodes'] = G.number_of_nodes()
    stats_dict['edges'] = G.number_of_edges()

    print("Computing connected components...")
    # Connected components
    strongly_cc = list(nx.strongly_connected_components(G))
    weakly_cc = list(nx.weakly_connected_components(G))

    stats_dict['num_strongly_cc'] = len(strongly_cc)
    stats_dict['size_largest_strongly_cc'] = len(max(strongly_cc, key=len))
    stats_dict['num_weakly_cc'] = len(weakly_cc)
    stats_dict['size_largest_weakly_cc'] = len(max(weakly_cc, key=len))

    # Density
    stats_dict['density'] = nx.density(G)

    # Giant component (largest weakly connected component)
    print("Analyzing giant component...")
    giant = G.subgraph(max(weakly_cc, key=len)).copy()
    G_undirected = giant.to_undirected()

    # Average clustering coefficient (approximated)
    print("Computing clustering coefficient...")
    try:
        if G_undirected.number_of_nodes() > 5000:
            # Sample for large graphs
            sample_nodes = list(G_undirected.nodes())[:min(5000, len(G_undirected.nodes()))]
            clustering_vals = [nx.clustering(G_undirected, node) for node in sample_nodes]
            stats_dict['avg_clustering'] = np.mean(clustering_vals)
        else:
            stats_dict['avg_clustering'] = nx.average_clustering(G_undirected)
    except Exception as e:
        print(f"Error computing clustering: {e}")
        stats_dict['avg_clustering'] = 'N/A'

    # Average distance (approximated using sampling)
    print("Computing average distance...")
    try:
        if G_undirected.number_of_nodes() > 1000:
            # Sample for large graphs
            sample_size = min(500, G_undirected.number_of_nodes())
            nodes_sample = list(G_undirected.nodes())[:sample_size]
            distances = []
            for node in nodes_sample:
                lengths = nx.single_source_shortest_path_length(G_undirected, node)
                distances.extend([d for d in lengths.values() if d > 0])
            stats_dict['avg_distance'] = np.mean(distances) if distances else 'N/A'
        else:
            if nx.is_connected(G_undirected):
                stats_dict['avg_distance'] = nx.average_shortest_path_length(G_undirected)
            else:
                # Calculate for largest component only
                largest_cc = max(nx.connected_components(G_undirected), key=len)
                G_largest = G_undirected.subgraph(largest_cc)
                stats_dict['avg_distance'] = nx.average_shortest_path_length(G_largest)
    except Exception as e:
        print(f"Error computing distance: {e}")
        stats_dict['avg_distance'] = 'N/A'

    return stats_dict, G, G_undirected


def plot_degree_distributions(G, output_prefix):
    """
    Plot indegree and outdegree distributions.
    Uses log-log scale with scatter plot to show tail behavior.
    """
    print("Plotting degree distributions...")
    indegrees = [d for n, d in G.in_degree()]
    outdegrees = [d for n, d in G.out_degree()]

    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

    # ========== Indegree Distribution ==========
    # Count degrees
    in_deg_count = defaultdict(int)
    for d in indegrees:
        in_deg_count[d] += 1

    if in_deg_count:
        degrees = sorted([k for k in in_deg_count.keys() if k > 0])
        counts = [in_deg_count[d] for d in degrees]

        # Plot 1: Log-log scatter plot (better for seeing tail)
        ax1.scatter(degrees, counts, alpha=0.6, s=20, c='blue')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Indegree (log scale)', fontsize=11)
        ax1.set_ylabel('Count (log scale)', fontsize=11)
        ax1.set_title('Indegree Distribution (log-log, scatter)', fontsize=12)
        ax1.grid(True, alpha=0.3, which='both')

        # Plot 2: Bar plot with log y-axis (alternative view)
        ax2.bar(range(len(degrees)), counts, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
        ax2.set_yscale('log')
        ax2.set_xlabel('Indegree', fontsize=11)
        ax2.set_ylabel('Count (log scale)', fontsize=11)
        ax2.set_title('Indegree Distribution (bar plot)', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')

        # Set x-axis ticks to show actual degree values
        if len(degrees) > 20:
            tick_positions = [0, len(degrees) // 4, len(degrees) // 2, 3 * len(degrees) // 4, len(degrees) - 1]
            tick_labels = [degrees[i] for i in tick_positions]
            ax2.set_xticks(tick_positions)
            ax2.set_xticklabels(tick_labels)

    # ========== Outdegree Distribution ==========
    out_deg_count = defaultdict(int)
    for d in outdegrees:
        out_deg_count[d] += 1

    if out_deg_count:
        degrees = sorted([k for k in out_deg_count.keys() if k > 0])
        counts = [out_deg_count[d] for d in degrees]

        # Plot 3: Log-log scatter plot
        ax3.scatter(degrees, counts, alpha=0.6, s=20, c='red')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Outdegree (log scale)', fontsize=11)
        ax3.set_ylabel('Count (log scale)', fontsize=11)
        ax3.set_title('Outdegree Distribution (log-log, scatter)', fontsize=12)
        ax3.grid(True, alpha=0.3, which='both')

        # Plot 4: Bar plot with log y-axis
        ax4.bar(range(len(degrees)), counts, alpha=0.7, color='red', edgecolor='black', linewidth=0.5)
        ax4.set_yscale('log')
        ax4.set_xlabel('Outdegree', fontsize=11)
        ax4.set_ylabel('Count (log scale)', fontsize=11)
        ax4.set_title('Outdegree Distribution (bar plot)', fontsize=12)
        ax4.grid(True, alpha=0.3, axis='y')

        # Set x-axis ticks
        if len(degrees) > 20:
            tick_positions = [0, len(degrees) // 4, len(degrees) // 2, 3 * len(degrees) // 4, len(degrees) - 1]
            tick_labels = [degrees[i] for i in tick_positions]
            ax4.set_xticks(tick_positions)
            ax4.set_xticklabels(tick_labels)

    plt.tight_layout()
    output_file = f'{output_prefix}_degree_distributions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_distance_distribution(G_undirected, output_prefix):
    """
    Plot distance distribution of giant component.
    Uses log y-axis to show tail behavior clearly.
    """
    print("Plotting distance distribution...")

    # Get largest connected component
    if not nx.is_connected(G_undirected):
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_undirected = G_undirected.subgraph(largest_cc).copy()

    # Sample if graph is too large
    if G_undirected.number_of_nodes() > 500:
        sample_size = min(300, G_undirected.number_of_nodes())
        nodes_sample = list(G_undirected.nodes())[:sample_size]
        distances = []
        for i, node in enumerate(nodes_sample):
            if i % 50 == 0:
                print(f"  Computing distances from node {i + 1}/{sample_size}...")
            lengths = nx.single_source_shortest_path_length(G_undirected, node)
            distances.extend([d for d in lengths.values() if d > 0])
    else:
        distances = []
        for node in G_undirected.nodes():
            lengths = nx.single_source_shortest_path_length(G_undirected, node)
            distances.extend([d for d in lengths.values() if d > 0])

    if not distances:
        print("No distances to plot")
        return

    dist_count = defaultdict(int)
    for d in distances:
        dist_count[d] += 1

    dists = sorted(dist_count.keys())
    counts = [dist_count[d] for d in dists]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Linear scale (original)
    ax1.bar(dists, counts, alpha=0.7, color='green', edgecolor='black')
    ax1.set_xlabel('Distance', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distance Distribution (linear scale)', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Log y-axis (to see tail)
    ax2.bar(dists, counts, alpha=0.7, color='green', edgecolor='black')
    ax2.set_yscale('log')
    ax2.set_xlabel('Distance', fontsize=12)
    ax2.set_ylabel('Count (log scale)', fontsize=12)
    ax2.set_title('Distance Distribution (log y-axis)', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = f'{output_prefix}_distance_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_distance_distribution(G_undirected, output_prefix):
    """
    Plot distance distribution of giant component.
    Uses log y-axis to show tail behavior clearly.
    """
    print("Plotting distance distribution...")

    # Get largest connected component
    if not nx.is_connected(G_undirected):
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_undirected = G_undirected.subgraph(largest_cc).copy()

    # Sample if graph is too large
    if G_undirected.number_of_nodes() > 500:
        sample_size = min(300, G_undirected.number_of_nodes())
        nodes_sample = list(G_undirected.nodes())[:sample_size]
        distances = []
        for i, node in enumerate(nodes_sample):
            if i % 50 == 0:
                print(f"  Computing distances from node {i + 1}/{sample_size}...")
            lengths = nx.single_source_shortest_path_length(G_undirected, node)
            distances.extend([d for d in lengths.values() if d > 0])
    else:
        distances = []
        for node in G_undirected.nodes():
            lengths = nx.single_source_shortest_path_length(G_undirected, node)
            distances.extend([d for d in lengths.values() if d > 0])

    if not distances:
        print("No distances to plot")
        return

    dist_count = defaultdict(int)
    for d in distances:
        dist_count[d] += 1

    dists = sorted(dist_count.keys())
    counts = [dist_count[d] for d in dists]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Linear scale (original)
    ax1.bar(dists, counts, alpha=0.7, color='green', edgecolor='black')
    ax1.set_xlabel('Distance', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distance Distribution (linear scale)', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Log y-axis (to see tail)
    ax2.bar(dists, counts, alpha=0.7, color='green', edgecolor='black')
    ax2.set_yscale('log')
    ax2.set_xlabel('Distance', fontsize=12)
    ax2.set_ylabel('Count (log scale)', fontsize=12)
    ax2.set_title('Distance Distribution (log y-axis)', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = f'{output_prefix}_distance_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ===================== Question 3.3 =====================

def compute_top_users_centrality(G, top_n=20):
    """
    Compute top users by different centrality measures.
    """
    print("Computing centrality measures...")

    # Degree centrality (using total degree = in + out)
    print("  Computing degree centrality...")
    total_degree = {node: G.in_degree(node) + G.out_degree(node)
                    for node in G.nodes()}
    top_degree = sorted(total_degree.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Betweenness centrality (on undirected version for tractability)
    print("  Computing betweenness centrality...")
    G_undirected = G.to_undirected()

    # Get largest connected component
    if not nx.is_connected(G_undirected):
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_undirected = G_undirected.subgraph(largest_cc).copy()

    if G_undirected.number_of_nodes() > 5000:
        # Use approximation for large graphs
        k_sample = min(1000, G_undirected.number_of_nodes() // 2)
        betweenness = nx.betweenness_centrality(G_undirected, k=k_sample)
    else:
        betweenness = nx.betweenness_centrality(G_undirected)
    top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Closeness centrality (on largest connected component)
    print("  Computing closeness centrality...")
    closeness = nx.closeness_centrality(G_undirected)
    top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return {
        'degree': top_degree,
        'betweenness': top_betweenness,
        'closeness': top_closeness
    }


def compare_rankings(rankings_dict, top_n=20):
    """
    Compare similarity of rankings using Kendall's tau and Jaccard similarity.
    """
    print("Comparing rankings...")

    # Get top-n users from each ranking
    sets = {name: set([user for user, _ in ranking[:top_n]])
            for name, ranking in rankings_dict.items()}

    # Jaccard similarity
    measures = list(rankings_dict.keys())
    jaccard_results = {}

    for i in range(len(measures)):
        for j in range(i + 1, len(measures)):
            m1, m2 = measures[i], measures[j]
            intersection = len(sets[m1] & sets[m2])
            union = len(sets[m1] | sets[m2])
            jaccard = intersection / union if union > 0 else 0
            jaccard_results[f'{m1}_vs_{m2}'] = jaccard

    # Kendall's tau on common users
    kendall_results = {}
    for i in range(len(measures)):
        for j in range(i + 1, len(measures)):
            m1, m2 = measures[i], measures[j]
            common_users = sets[m1] & sets[m2]

            if len(common_users) > 1:
                rank1 = {user: rank for rank, (user, _) in enumerate(rankings_dict[m1][:top_n])}
                rank2 = {user: rank for rank, (user, _) in enumerate(rankings_dict[m2][:top_n])}

                ranks1 = [rank1[user] for user in common_users if user in rank1]
                ranks2 = [rank2[user] for user in common_users if user in rank2]

                if len(ranks1) > 1:
                    tau, _ = stats.kendalltau(ranks1, ranks2)
                    kendall_results[f'{m1}_vs_{m2}'] = tau

    return jaccard_results, kendall_results


# ===================== Question 3.4 =====================

def detect_communities(G_undirected):
    """
    Detect communities in the graph.
    """
    print("Detecting communities...")

    # Get largest connected component
    if not nx.is_connected(G_undirected):
        giant_cc = max(nx.connected_components(G_undirected), key=len)
        G_giant = G_undirected.subgraph(giant_cc).copy()
    else:
        G_giant = G_undirected.copy()

    print(f"  Working with giant component of {G_giant.number_of_nodes()} nodes")

    # Try Louvain first, fallback to greedy modularity
    try:
        import community as community_louvain
        print("  Using Louvain algorithm...")
        communities = community_louvain.best_partition(G_giant)
        modularity = community_louvain.modularity(communities, G_giant)
    except ImportError:
        print("  Using greedy modularity algorithm...")
        from networkx.algorithms import community
        communities_sets = community.greedy_modularity_communities(G_giant)
        communities = {}
        for idx, comm in enumerate(communities_sets):
            for node in comm:
                communities[node] = idx
        modularity = community.modularity(G_giant, communities_sets)

    # Get community sizes
    comm_sizes = defaultdict(int)
    for node, comm_id in communities.items():
        comm_sizes[comm_id] += 1

    return communities, modularity, dict(comm_sizes)


# ===================== Question 3.5 =====================

def plot_weight_distribution(edge_list_file, output_prefix):
    """
    Plot weight distribution of edges.
    """
    print("Plotting weight distribution...")

    weights = []
    with open(edge_list_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            weights.append(int(row['weight']))

    weight_count = defaultdict(int)
    for w in weights:
        weight_count[w] += 1

    ws = sorted(weight_count.keys())
    counts = [weight_count[w] for w in ws]

    # Plot log-log
    plt.figure(figsize=(10, 6))
    plt.loglog(ws, counts, 'bo', alpha=0.6, markersize=4)
    plt.xlabel('Weight', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Weight Distribution (log-log scale)', fontsize=14)
    plt.grid(True, alpha=0.3)
    output_file = f'{output_prefix}_weight_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


# ===================== Main Execution =====================

def main():
    # Question 3.1 & 3.2 - Small dataset
    print("=" * 60)
    print("Processing twitter-small.tsv")
    print("=" * 60)

    input_file_small = os.path.join(DATA_DIR, 'twitter-small.tsv')
    output_file_small = os.path.join(OUTPUT_DIR, 'mention_graph_small.csv')

    edge_list_small = parse_twitter_data(input_file_small, output_file_small)
    stats_small, G_small, G_small_undirected = compute_network_statistics(edge_list_small)

    print("\nNetwork Statistics (Small Dataset):")
    for key, value in stats_small.items():
        print(f"  {key}: {value}")

    plot_degree_distributions(G_small, os.path.join(OUTPUT_DIR, 'small'))
    plot_distance_distribution(G_small_undirected, os.path.join(OUTPUT_DIR, 'small'))

    # Question 3.3
    print("\n" + "=" * 60)
    print("Computing centrality measures (Small Dataset)")
    print("=" * 60)

    top_users_small = compute_top_users_centrality(G_small)
    for measure, users in top_users_small.items():
        print(f"\nTop 20 by {measure}:")
        for i, (user, score) in enumerate(users[:10], 1):
            print(f"  {i}. {user}: {score:.6f}")

    jaccard_small, kendall_small = compare_rankings(top_users_small)
    print("\nRanking Similarities (Jaccard):")
    for pair, score in jaccard_small.items():
        print(f"  {pair}: {score:.4f}")

    print("\nRanking Similarities (Kendall's tau):")
    for pair, score in kendall_small.items():
        print(f"  {pair}: {score:.4f}")

    # Question 3.4
    print("\n" + "=" * 60)
    print("Community Detection (Small Dataset)")
    print("=" * 60)

    communities_small, modularity_small, sizes_small = detect_communities(G_small_undirected)
    print(f"Number of communities: {len(sizes_small)}")
    print(f"Modularity: {modularity_small:.4f}")
    print(f"Top 10 community sizes: {sorted(sizes_small.values(), reverse=True)[:10]}")

    # Question 3.5
    print("\n" + "=" * 60)
    print("Weight Distribution (Small Dataset)")
    print("=" * 60)

    plot_weight_distribution(edge_list_small, os.path.join(OUTPUT_DIR, 'small'))

    # Question 3.6 - Larger dataset
    print("\n" + "=" * 60)
    print("Processing twitter-larger.tsv")
    print("=" * 60)

    input_file_larger = os.path.join(DATA_DIR, 'twitter-larger.tsv')
    output_file_larger = os.path.join(OUTPUT_DIR, 'mention_graph_larger.csv')

    edge_list_larger = parse_twitter_data(input_file_larger, output_file_larger)
    stats_larger, G_larger, G_larger_undirected = compute_network_statistics(edge_list_larger)

    print("\nNetwork Statistics (Larger Dataset):")
    for key, value in stats_larger.items():
        print(f"  {key}: {value}")

    plot_degree_distributions(G_larger, os.path.join(OUTPUT_DIR, 'larger'))
    plot_distance_distribution(G_larger_undirected, os.path.join(OUTPUT_DIR, 'larger'))

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()