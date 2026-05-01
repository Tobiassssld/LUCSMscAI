"""
OddBall Core Implementation for SNACS Course Paper
Clean, importable functions without test code

Usage:
    from oddball_core import oddball
    
    scores, details = oddball(G, verbose=True)
    # scores: dict {node: anomaly_score}
    # details: dict {node: {feature: value}}
"""

import numpy as np
import networkx as nx
from numpy.linalg import eigvals


# ============================================================================
# Core OddBall Implementation
# ============================================================================

def fit_powerlaw_robust(X, Y, outlier_percentile=95):
    """
    Robust power-law fitting: Y = C * X^a
    Removes outliers before fitting to improve robustness.
    
    Args:
        X: array-like, independent variable
        Y: array-like, dependent variable
        outlier_percentile: percentile threshold for outlier removal (default: 95)
    
    Returns:
        C: multiplicative constant
        a: power-law exponent
    """
    X = np.array(X)
    Y = np.array(Y)
    
    # Filter invalid values
    mask = (X > 0) & (Y > 0) & np.isfinite(X) & np.isfinite(Y)
    X = X[mask]
    Y = Y[mask]
    
    if len(X) < 2:
        return 1.0, 1.0
    
    # Transform to log-log space
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
    
    # Second pass: fit on inliers only
    slope, intercept = np.polyfit(logX[inlier_mask], logY[inlier_mask], 1)
    C = np.exp(intercept)
    
    return C, slope


def oddball_score_original(y_true, y_pred, epsilon=1e-10):
    """
    Original OddBall scoring formula from Akoglu et al. (2010).
    
    Formula: out-line(i) = max(y, ŷ) / min(y, ŷ) * log(|y - ŷ| + 1)
    
    Args:
        y_true: actual observed value
        y_pred: predicted value from power-law
        epsilon: small constant to avoid division by zero
    
    Returns:
        anomaly score (higher = more anomalous)
    """
    y_true = max(y_true, epsilon)
    y_pred = max(y_pred, epsilon)
    
    ratio = max(y_true, y_pred) / min(y_true, y_pred)
    diff = np.log(abs(y_true - y_pred) + 1.0)
    
    return ratio * diff


def get_egonet(G, v):
    """
    Extract egonet: node v + all neighbors + edges between them.
    
    Args:
        G: NetworkX graph
        v: node identifier
    
    Returns:
        egonet: NetworkX subgraph
    """
    neighbors = set(G.neighbors(v))
    nodes = neighbors | {v}
    return G.subgraph(nodes)


def egonet_features(G, egonet):
    """
    Extract egonet features as per OddBall paper.
    
    Features:
        N: number of nodes (neighbors + ego)
        E: number of edges in egonet
        W: sum of edge weights
        λw: principal eigenvalue of weighted adjacency matrix
    
    Args:
        G: NetworkX graph (needed for weight attribute)
        egonet: NetworkX subgraph (the egonet)
    
    Returns:
        (N, E, W, λw): tuple of features
    """
    N = egonet.number_of_nodes()
    E = egonet.number_of_edges()
    
    # Sum edge weights
    W = sum(data.get("weight", 1.0) for _, _, data in egonet.edges(data=True))
    
    # Compute principal eigenvalue
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
    OddBall anomaly detection algorithm (Akoglu et al., 2010).
    
    Detects three types of anomalies:
        - CliqueStar: nodes with unusual ego-network density (E vs N)
        - HeavyVicinity: nodes with unusual total weight (W vs E)
        - DominantPair: nodes with unusual principal eigenvalue (λw vs W)
    
    Args:
        G: NetworkX graph (must have 'weight' attribute on edges)
        verbose: if True, print diagnostic information
    
    Returns:
        scores: dict {node: combined_anomaly_score}
        details: dict {node: {
            'cliqueStar': score,
            'heavyVicinity': score,
            'dominantPair': score,
            'N': num_nodes,
            'E': num_edges,
            'W': total_weight,
            'lambda_w': principal_eigenvalue,
            'E_pred': predicted_E,
            'W_pred': predicted_W,
            'L_pred': predicted_lambda
        }}
    
    Example:
        >>> import networkx as nx
        >>> from oddball_v2 import oddball
        >>> 
        >>> # Create a graph
        >>> G = nx.karate_club_graph()
        >>> for u, v in G.edges():
        >>>     G[u][v]['weight'] = 1.0
        >>> 
        >>> # Run OddBall
        >>> scores, details = oddball(G, verbose=True)
        >>> 
        >>> # Get top anomalies
        >>> top_10 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        >>> for node, score in top_10:
        >>>     print(f"Node {node}: score={score:.2f}")
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
    
    # Fit power-laws for the 3 OddBall detectors
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
# Convenience functions
# ============================================================================

def get_top_anomalies(scores, k=10):
    """
    Get the top-k anomalies by score.
    
    Args:
        scores: dict {node: score}
        k: number of top anomalies to return
    
    Returns:
        list of (node, score) tuples, sorted by score (descending)
    """
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def print_top_anomalies(scores, details, k=10):
    """
    Print the top-k anomalies with their features.
    
    Args:
        scores: dict {node: score}
        details: dict {node: feature_dict}
        k: number of top anomalies to print
    """
    print(f"\nTop {k} Anomalies:")
    print(f"{'Rank':<6} {'Node':<10} {'Score':<12} {'N':<6} {'E':<8} {'W':<10}")
    print("-" * 60)
    
    top = get_top_anomalies(scores, k)
    for rank, (node, score) in enumerate(top, 1):
        d = details[node]
        print(f"{rank:<6} {node:<10} {score:<12.4f} {d['N']:<6} {d['E']:<8} {d['W']:<10.1f}")


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    print("OddBall")
    print("=" * 70)
    print("Import this module to use OddBall in your code:")
    print()
    print("    from oddball_core import oddball")
    print("    scores, details = oddball(G, verbose=True)")
    print()
    print("=" * 70)
    
    # Quick demo on Karate Club graph
    print("\nQuick demo on Karate Club graph:")
    G = nx.karate_club_graph()
    
    # Add weights
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
    
    scores, details = oddball(G, verbose=True)
    print_top_anomalies(scores, details, k=5)
