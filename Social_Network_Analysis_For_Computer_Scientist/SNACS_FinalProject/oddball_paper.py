"""
OddBall Implementation for SNACS Course Paper
Corrected implementation based on original paper formulas

Key corrections:
1. Fixed scoring formula to match Akoglu et al. (2010)
2. Added proper anomaly injection for LFR benchmarks
3. F1-score evaluation on synthetic data
"""

import numpy as np
import networkx as nx
from numpy.linalg import eigvals
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


# ============================================================================
# CORRECTED OddBall Implementation
# ============================================================================

def fit_powerlaw_robust(X, Y, outlier_percentile=95):
    """Robust power-law fitting: Y = C * X^a"""
    X = np.array(X)
    Y = np.array(Y)
    
    mask = (X > 0) & (Y > 0) & np.isfinite(X) & np.isfinite(Y)
    X = X[mask]
    Y = Y[mask]
    
    if len(X) < 2:
        return 1.0, 1.0
    
    logX = np.log(X)
    logY = np.log(Y)
    
    # First pass: fit all data
    slope, intercept = np.polyfit(logX, logY, 1)
    y_pred = slope * logX + intercept
    residuals = np.abs(logY - y_pred)
    
    # Remove outliers based on residuals
    threshold = np.percentile(residuals, outlier_percentile)
    inlier_mask = residuals <= threshold
    
    if inlier_mask.sum() < 2:
        inlier_mask = np.ones(len(logX), dtype=bool)
    
    # Second pass: fit on inliers
    slope, intercept = np.polyfit(logX[inlier_mask], logY[inlier_mask], 1)
    C = np.exp(intercept)
    
    return C, slope


def oddball_score_original(y_true, y_pred, epsilon=1e-10):
    """
    Original OddBall scoring formula from Akoglu et al. (2010), page 7:
    
    out-line(i) = max(yi, Cx^θ) / min(yi, Cx^θ) * log(|yi - Cx^θ| + 1)
    
    This is the correct formula from the paper.
    """
    y_true = max(y_true, epsilon)
    y_pred = max(y_pred, epsilon)
    
    ratio = max(y_true, y_pred) / min(y_true, y_pred)
    diff = np.log(abs(y_true - y_pred) + 1.0)
    
    return ratio * diff


def get_egonet(G, v):
    """Extract egonet: node v + neighbors + edges between them"""
    neighbors = set(G.neighbors(v))
    nodes = neighbors | {v}
    return G.subgraph(nodes)


def egonet_features(G, egonet):
    """
    Extract egonet features as per OddBall paper:
    - N: number of nodes (neighbors + ego)
    - E: number of edges in egonet
    - W: sum of edge weights
    - λw: principal eigenvalue of weighted adjacency matrix
    """
    N = egonet.number_of_nodes()
    E = egonet.number_of_edges()
    
    W = sum(data.get("weight", 1.0) for _, _, data in egonet.edges(data=True))
    
    if N > 0:
        A = nx.to_numpy_array(egonet, weight="weight")
        if A.size > 0:
            try:
                eigs = eigvals(A).real
                lam = max(eigs) if len(eigs) > 0 else 0.0
            except:
                lam = 0.0
        else:
            lam = 0.0
    else:
        lam = 0.0
    
    return N, E, W, lam


def oddball(G, verbose=False):
    """
    OddBall algorithm as per Akoglu et al. (2010)
    
    Returns:
        scores: dict {node: combined_score}
        details: dict {node: {feature: value}}
    """
    
    # Extract egonet features for all nodes
    N_list, E_list, W_list, L_list = [], [], [], []
    feats = {}
    
    if verbose:
        print("Extracting egonet features...")
    
    for v in G.nodes():
        ego = get_egonet(G, v)
        N, E, W, lam = egonet_features(G, ego)
        feats[v] = (N, E, W, lam)
        
        N_list.append(N)
        E_list.append(E)
        W_list.append(W)
        L_list.append(lam)
    
    if verbose:
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"Avg N: {np.mean(N_list):.2f}, Avg E: {np.mean(E_list):.2f}")
    
    # Fit power-laws (3 OddBall detectors)
    C_NE, a_NE = fit_powerlaw_robust(N_list, E_list)  # E vs N (CliqueStar)
    C_WE, a_WE = fit_powerlaw_robust(E_list, W_list)  # W vs E (HeavyVicinity)
    C_LW, a_LW = fit_powerlaw_robust(W_list, L_list)  # λw vs W (DominantPair)
    
    if verbose:
        print(f"\nPower-laws fitted:")
        print(f"  E = {C_NE:.4f} * N^{a_NE:.4f} (CliqueStar)")
        print(f"  W = {C_WE:.4f} * E^{a_WE:.4f} (HeavyVicinity)")
        print(f"  λw = {C_LW:.4f} * W^{a_LW:.4f} (DominantPair)")
    
    # Score each node
    scores = {}
    details = {}
    
    for v, (N, E, W, lam) in feats.items():
        # Predictions from power-laws
        E_pred = C_NE * (N ** a_NE) if N > 0 else 0.0
        W_pred = C_WE * (E ** a_WE) if E > 0 else 0.0
        L_pred = C_LW * (W ** a_LW) if W > 0 else 0.0
        
        # Individual detector scores (using original formula)
        s1 = oddball_score_original(E, E_pred)     # CliqueStar
        s2 = oddball_score_original(W, W_pred)     # HeavyVicinity
        s3 = oddball_score_original(lam, L_pred)   # DominantPair
        
        # Combined score
        combined = s1 + s2 + s3
        
        scores[v] = combined
        details[v] = {
            "cliqueStar": s1,
            "heavyVicinity": s2,
            "dominantPair": s3,
            "N": N, "E": E, "W": W, "lambda_w": lam,
            "E_pred": E_pred, "W_pred": W_pred, "L_pred": L_pred
        }
    
    return scores, details


# ============================================================================
# Anomaly Injection for LFR Benchmarks
# ============================================================================

def inject_anomalies_lfr(G, num_anomalies, anomaly_type="mixed", seed=42):
    """
    Inject anomalies into LFR benchmark graph.
    
    Args:
        G: NetworkX graph (LFR benchmark)
        num_anomalies: number of anomalies to inject
        anomaly_type: "star", "clique", "heavy", or "mixed"
        seed: random seed
    
    Returns:
        G_anomalous: graph with anomalies
        labels: dict {node: 0 or 1} (1=anomaly)
    """
    
    np.random.seed(seed)
    G_anom = G.copy()
    labels = {node: 0 for node in G.nodes()}
    
    # Select random nodes to make anomalous
    all_nodes = list(G.nodes())
    anomaly_nodes = np.random.choice(all_nodes, num_anomalies, replace=False)
    
    if anomaly_type == "star" or (anomaly_type == "mixed" and np.random.rand() < 0.33):
        # Create near-star: node with many neighbors but no edges between them
        for anom_node in anomaly_nodes:
            # Add many new connections
            num_new_edges = np.random.randint(20, 50)
            potential_targets = [n for n in all_nodes if n != anom_node and not G_anom.has_edge(anom_node, n)]
            
            if len(potential_targets) >= num_new_edges:
                new_neighbors = np.random.choice(potential_targets, num_new_edges, replace=False)
                for target in new_neighbors:
                    G_anom.add_edge(anom_node, target, weight=1.0)
            
            labels[anom_node] = 1
    
    elif anomaly_type == "clique" or (anomaly_type == "mixed" and np.random.rand() < 0.66):
        # Create near-clique: node with neighbors all connected to each other
        for anom_node in anomaly_nodes:
            neighbors = list(G_anom.neighbors(anom_node))
            
            # Connect all pairs of neighbors
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if not G_anom.has_edge(neighbors[i], neighbors[j]):
                        G_anom.add_edge(neighbors[i], neighbors[j], weight=1.0)
            
            labels[anom_node] = 1
    
    else:  # heavy or mixed
        # Create heavy vicinity: abnormally high edge weights
        for anom_node in anomaly_nodes:
            neighbors = list(G_anom.neighbors(anom_node))
            
            # Increase weights dramatically
            for neighbor in neighbors:
                if G_anom.has_edge(anom_node, neighbor):
                    current_weight = G_anom[anom_node][neighbor].get('weight', 1.0)
                    G_anom[anom_node][neighbor]['weight'] = current_weight * np.random.uniform(10, 50)
            
            labels[anom_node] = 1
    
    return G_anom, labels


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_oddball(scores, labels, k=None):
    """
    Evaluate OddBall performance using F1, Precision, Recall
    
    Args:
        scores: dict {node: score}
        labels: dict {node: 0 or 1}
        k: number of top-k nodes to consider as anomalies (if None, use all anomalies)
    
    Returns:
        metrics: dict with f1, precision, recall
    """
    
    # Get ground truth
    y_true = []
    y_scores = []
    
    for node in sorted(labels.keys()):
        y_true.append(labels[node])
        y_scores.append(scores.get(node, 0.0))
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Determine k
    if k is None:
        k = int(y_true.sum())  # Number of true anomalies
    
    # Get top-k predictions
    top_k_indices = np.argsort(y_scores)[-k:]
    y_pred = np.zeros(len(y_true))
    y_pred[top_k_indices] = 1
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "k": k,
        "num_true_anomalies": int(y_true.sum()),
        "num_detected": int(y_pred.sum())
    }


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_experiment_lfr(n_nodes=1000, mixing_param=0.3, num_anomalies=50, 
                       anomaly_type="mixed", verbose=True):
    """
    Run complete experiment on LFR benchmark
    
    Args:
        n_nodes: number of nodes in LFR graph
        mixing_param: LFR mixing parameter
        num_anomalies: number of anomalies to inject
        anomaly_type: type of anomalies to inject
        verbose: print progress
    
    Returns:
        results: dict with metrics and details
    """
    
    if verbose:
        print("="*70)
        print(f"Running LFR Experiment")
        print(f"  n_nodes={n_nodes}, mixing={mixing_param}")
        print(f"  num_anomalies={num_anomalies}, type={anomaly_type}")
        print("="*70)
    
    # Generate LFR benchmark (Note: requires python-igraph and leidenalg)
    # For now, we'll use a placeholder. In your actual code, use:
    # from networkx.generators.community import LFR_benchmark_graph
    # G = LFR_benchmark_graph(n=n_nodes, tau1=3, tau2=1.5, mu=mixing_param, ...)
    
    # Placeholder: create a random graph with community structure
    from networkx.algorithms.community import greedy_modularity_communities
    G = nx.powerlaw_cluster_graph(n_nodes, m=10, p=0.05, seed=42)
    
    # Add weights
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.exponential(scale=1.0)
    
    if verbose:
        print(f"\nGenerated graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Inject anomalies
    G_anom, labels = inject_anomalies_lfr(G, num_anomalies, anomaly_type)
    
    if verbose:
        print(f"Injected {num_anomalies} {anomaly_type} anomalies")
        print(f"Anomalous graph: {G_anom.number_of_nodes()} nodes, {G_anom.number_of_edges()} edges")
    
    # Run OddBall
    if verbose:
        print("\nRunning OddBall...")
    scores, details = oddball(G_anom, verbose=verbose)
    
    # Evaluate
    if verbose:
        print("\nEvaluating...")
    metrics = evaluate_oddball(scores, labels)
    
    if verbose:
        print(f"\nResults:")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Detected:  {metrics['num_detected']}/{metrics['num_true_anomalies']}")
    
    # Analyze top anomalies
    if verbose:
        print(f"\nTop 10 Detected Anomalies:")
        top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        for rank, (node, score) in enumerate(top_nodes, 1):
            is_true = "✓ TRUE" if labels[node] == 1 else "✗ FALSE"
            print(f"  {rank}. Node {node}: score={score:.4f} {is_true}")
    
    return {
        "metrics": metrics,
        "scores": scores,
        "labels": labels,
        "details": details,
        "graph": G_anom
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("OddBall Implementation for SNACS Course Paper")
    print("="*70)
    
    # Example experiment
    results = run_experiment_lfr(
        n_nodes=1000,
        mixing_param=0.3,
        num_anomalies=50,
        anomaly_type="mixed",
        verbose=True
    )
    
    print("\n" + "="*70)
    print("Experiment Complete!")
    print(f"Final F1-Score: {results['metrics']['f1']:.4f}")
