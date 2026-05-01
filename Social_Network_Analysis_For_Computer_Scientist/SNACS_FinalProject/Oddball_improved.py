"""
Improved Weighted OddBall Implementation for SNACS final project

Key improvements:
1. Fixed scoring formula to match original OddBall paper
2. Added robust power-law fitting with outlier handling
3. Better handling of zero/small values
4. Added normalization options
"""

import numpy as np
import networkx as nx
from numpy.linalg import eigvals


# --------------------------------------------------
# Utility: Robust power-law fit
# --------------------------------------------------

def fit_powerlaw_robust(X, Y, outlier_percentile=95):
    """
    Fit Y = C * X^a using log-log linear regression with outlier removal.
    Returns (C, a).
    """
    X = np.array(X)
    Y = np.array(Y)

    # Filter out invalid values
    mask = (X > 0) & (Y > 0) & np.isfinite(X) & np.isfinite(Y)
    X = X[mask]
    Y = Y[mask]

    if len(X) < 2:
        return 1.0, 1.0

    logX = np.log(X)
    logY = np.log(Y)
    
    # First pass: fit to get residuals
    slope, intercept = np.polyfit(logX, logY, 1)
    y_pred = slope * logX + intercept
    residuals = np.abs(logY - y_pred)
    
    # Remove outliers (points with large residuals)
    threshold = np.percentile(residuals, outlier_percentile)
    inlier_mask = residuals <= threshold
    
    if inlier_mask.sum() < 2:
        # If too few inliers, use all data
        inlier_mask = np.ones(len(logX), dtype=bool)
    
    # Second pass: fit on inliers only
    slope, intercept = np.polyfit(logX[inlier_mask], logY[inlier_mask], 1)
    C = np.exp(intercept)
    
    return C, slope


# --------------------------------------------------
# OddBall scoring - Original Paper Formula
# --------------------------------------------------

def oddball_score_original(y_true, y_pred, epsilon=1e-10):
    """
    Original OddBall scoring formula from the paper:
        O(i) = |y - ŷ| / min(y, ŷ)
    
    This measures the relative deviation from the expected value.
    Higher score = more anomalous
    """
    y_true = max(y_true, epsilon)
    y_pred = max(y_pred, epsilon)
    
    numerator = abs(y_true - y_pred)
    denominator = min(y_true, y_pred)
    
    return numerator / denominator


def oddball_score_modified(y_true, y_pred, epsilon=1e-10):
    """
    Modified scoring that handles zero values better and adds log scaling.
        O(i) = max(y, ŷ) / min(y, ŷ) * log(|y - ŷ| + 1)
    
    This version amplifies large deviations more than the original.
    """
    y_true = max(y_true, epsilon)
    y_pred = max(y_pred, epsilon)
    
    ratio = max(y_true, y_pred) / min(y_true, y_pred)
    diff = np.log(abs(y_true - y_pred) + 1.0)
    
    return ratio * diff


# --------------------------------------------------
# Egonet feature extraction
# --------------------------------------------------

def get_egonet(G, v):
    """Get egonet: node v + all neighbors + edges between them"""
    neighbors = set(G.neighbors(v))
    nodes = neighbors | {v}
    return G.subgraph(nodes)


def egonet_features(G, egonet):
    """
    Returns:
        N  = number of nodes in egonet (neighbors + self)
        E  = number of edges in egonet
        W  = sum of edge weights
        λ1 = largest eigenvalue of weighted adjacency matrix
    """
    N = egonet.number_of_nodes()
    E = egonet.number_of_edges()

    # Weighted sum
    W = 0.0
    for u, v, data in egonet.edges(data=True):
        weight = data.get("weight", 1.0)
        W += weight

    # Weighted adjacency matrix for λ1
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


# --------------------------------------------------
# Main OddBall function
# --------------------------------------------------

def oddball(G, use_original_formula=True, normalize=False, verbose=False):
    """
    Compute OddBall anomaly scores for a weighted graph.
    
    Args:
        G: NetworkX graph
        use_original_formula: If True, use original paper formula; 
                            if False, use modified formula
        normalize: If True, normalize scores to [0, 1] range
        verbose: Print diagnostic information
    
    Returns:
        scores[node] = combined anomaly score
        details[node] = {"cliqueStar":..., "heavyVicinity":..., "dominantPair":...}
    """
    
    # Choose scoring function
    score_func = oddball_score_original if use_original_formula else oddball_score_modified

    # First pass: collect per-node egonet features
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
        print(f"Graph statistics:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Avg N: {np.mean(N_list):.2f}")
        print(f"  Avg E: {np.mean(E_list):.2f}")
        print(f"  Avg W: {np.mean(W_list):.2f}")
        print(f"  Avg λ1: {np.mean(L_list):.2f}")

    # Fit power-laws for the 3 OddBall relations using robust fitting
    if verbose:
        print("\nFitting power-laws...")
    
    C_NE, a_NE = fit_powerlaw_robust(N_list, E_list)      # E vs N
    C_WE, a_WE = fit_powerlaw_robust(E_list, W_list)      # W vs E  
    C_LW, a_LW = fit_powerlaw_robust(W_list, L_list)      # λ1 vs W

    if verbose:
        print(f"  E = {C_NE:.4f} * N^{a_NE:.4f}")
        print(f"  W = {C_WE:.4f} * E^{a_WE:.4f}")
        print(f"  λ1 = {C_LW:.4f} * W^{a_LW:.4f}")

    scores = {}
    details = {}

    # Second pass: score each node
    if verbose:
        print("\nComputing anomaly scores...")
    
    for v, (N, E, W, lam) in feats.items():

        # Predict values from power-laws
        E_pred  = C_NE * (N ** a_NE) if N > 0 else 0.0
        W_pred  = C_WE * (E ** a_WE) if E > 0 else 0.0
        L_pred  = C_LW * (W ** a_LW) if W > 0 else 0.0

        # Compute OddBall anomaly scores
        s1 = score_func(E, E_pred)      # CliqueStar
        s2 = score_func(W, W_pred)      # HeavyVicinity
        s3 = score_func(lam, L_pred)    # DominantPair

        combined = s1 + s2 + s3

        scores[v] = combined
        details[v] = {
            "cliqueStar": s1,
            "heavyVicinity": s2,
            "dominantPair": s3,
            "N": N, "E": E, "W": W, "lambda1": lam,
            "E_pred": E_pred, "W_pred": W_pred, "L_pred": L_pred
        }

    # Optional normalization
    if normalize and len(scores) > 0:
        max_score = max(scores.values())
        if max_score > 0:
            scores = {k: v/max_score for k, v in scores.items()}

    return scores, details


# --------------------------------------------------
# Additional utility: Analyze score distribution
# --------------------------------------------------

def analyze_scores(scores, details, labels=None, top_k=20):
    """
    Analyze the distribution of anomaly scores.
    
    Args:
        scores: dict of {node: score}
        details: dict of {node: detailed scores}
        labels: optional ground truth labels (1=anomaly, 0=normal)
        top_k: number of top anomalies to display
    """
    print("\n" + "="*60)
    print("Score Distribution Analysis")
    print("="*60)
    
    score_array = np.array(list(scores.values()))
    print(f"Total nodes scored: {len(score_array)}")
    print(f"Mean score: {np.mean(score_array):.4f}")
    print(f"Median score: {np.median(score_array):.4f}")
    print(f"Std score: {np.std(score_array):.4f}")
    print(f"Min score: {np.min(score_array):.4f}")
    print(f"Max score: {np.max(score_array):.4f}")
    
    # Percentiles
    percentiles = [50, 75, 90, 95, 99]
    print(f"\nScore percentiles:")
    for p in percentiles:
        val = np.percentile(score_array, p)
        print(f"  {p}th: {val:.4f}")
    
    # Top anomalies
    print(f"\nTop {top_k} anomalies:")
    print(f"{'Node':<10} {'Score':<10} {'CS':<8} {'HV':<8} {'DP':<8} {'Label':<10}")
    print("-"*60)
    
    top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    for node, score in top_nodes:
        d = details[node]
        label_str = "FRAUD" if labels is not None and labels[node] == 1 else "Normal"
        print(f"{node:<10} {score:<10.4f} {d['cliqueStar']:<8.4f} "
              f"{d['heavyVicinity']:<8.4f} {d['dominantPair']:<8.4f} {label_str:<10}")
    
    # If labels provided, analyze true positives
    if labels is not None:
        fraud_nodes = [i for i, label in enumerate(labels) if label == 1]
        fraud_scores = [scores.get(i, 0) for i in fraud_nodes]
        normal_nodes = [i for i, label in enumerate(labels) if label == 0]
        normal_scores = [scores.get(i, 0) for i in normal_nodes]
        
        print(f"\n Fraud nodes: {len(fraud_nodes)}")
        print(f"  Mean fraud score: {np.mean(fraud_scores):.4f}")
        print(f"  Median fraud score: {np.median(fraud_scores):.4f}")
        
        print(f"\nNormal nodes: {len(normal_nodes)}")
        print(f"  Mean normal score: {np.mean(normal_scores):.4f}")
        print(f"  Median normal score: {np.median(normal_scores):.4f}")
        
        # Check if fraud scores are higher
        if np.mean(fraud_scores) > np.mean(normal_scores):
            print("\n✓ Fraud nodes have higher average scores (good!)")
        else:
            print("\n✗ Fraud nodes have LOWER average scores (problem!)")


# --------------------------------------------------
# Main method
# --------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to weighted edge list (u v w)")
    parser.add_argument("--formula", type=str, default="original",
                        choices=["original", "modified"],
                        help="Scoring formula to use")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize scores to [0,1]")
    parser.add_argument("--verbose", action="store_true",
                        help="Print diagnostic information")
    args = parser.parse_args()

    print("Loading graph...")
    G = nx.read_weighted_edgelist(args.input)

    print("Running OddBall Algorithm...")
    use_original = (args.formula == "original")
    scores, details = oddball(G, 
                             use_original_formula=use_original,
                             normalize=args.normalize,
                             verbose=args.verbose)

    print(f"\nTop 20 anomalies (using {args.formula} formula):")
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
    for node, score in top:
        print(f"{node}\t{score:.4f}")
