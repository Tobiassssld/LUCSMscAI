"""
Weighted OddBall Implemenetation for SNACS final project_20251127

Implements the three main OddBall anomaly detectors:
    - CliqueStar      : E vs N (Egonet Density Power Law)
    - HeavyVicinity   : W vs E (Weight Power Law)
    - DominantPair    : λ1 vs W (Weighted Eigenvalue Power Law)
"""

import numpy as np
import networkx as nx
from numpy.linalg import eigvals


# --------------------------------------------------
# Utility: log-log power-law fit
# --------------------------------------------------

def fit_powerlaw(X, Y):
    """
    Fit Y = C * X^a using log-log linear regression.
    Returns (C, a).
    """
    X = np.array(X)
    Y = np.array(Y)

    mask = (X > 0) & (Y > 0)
    X = X[mask]
    Y = Y[mask]

    if len(X) < 2:
        return 1.0, 1.0

    logX = np.log(X)
    logY = np.log(Y)
    slope, intercept = np.polyfit(logX, logY, 1)
    C = np.exp(intercept)
    return C, slope


# --------------------------------------------------
# OddBall scoring core
# --------------------------------------------------

def oddball_score(y_true, y_pred):
    """
    OddBall scoring formula:
        out(i) = max(y, y_pred) / min(y, y_pred) * log(|y - y_pred| + 1)
    """
    if y_true <= 0 or y_pred <= 0:
        return 0.0
    ratio = max(y_true, y_pred) / min(y_true, y_pred)
    diff = np.log(abs(y_true - y_pred) + 1.0)
    return ratio * diff


# --------------------------------------------------
# Egonet feature extraction
# --------------------------------------------------

def get_egonet(G, v):
    neighbors = list(G.neighbors(v))
    nodes = neighbors + [v]
    return G.subgraph(nodes)


def egonet_features(G, egonet):
    """
    Returns:
        N  = number of nodes in egonet
        E  = number of edges in egonet
        W  = sum of edge weights
        λ1 = largest eigenvalue of weighted adjacency matrix
    """
    N = egonet.number_of_nodes()
    E = egonet.number_of_edges()

    # Weighted sum
    W = 0
    for u, w, data in egonet.edges(data=True):
        weight = data.get("weight", 1.0)
        W += weight

    # Weighted adjacency matrix for λ1
    A = nx.to_numpy_array(egonet, weight="weight")
    if A.size == 0:
        lam = 0
    else:
        try:
            lam = max(eigvals(A).real)
        except:
            lam = 0

    return N, E, W, lam


# --------------------------------------------------
# Main OddBall function
# --------------------------------------------------

def oddball(G):
    """
    Compute full OddBall anomaly scores for a weighted graph.
    Returns:
        scores[node] = combined anomaly score
        details[node] = {"cliqueStar":..., "heavyVicinity":..., "dominantPair":...}
    """

    # First pass: collect per-node egonet features
    N_list, E_list, W_list, L_list = [], [], [], []
    feats = {}

    for v in G.nodes():
        ego = get_egonet(G, v)
        N, E, W, lam = egonet_features(G, ego)
        feats[v] = (N, E, W, lam)

        N_list.append(N)
        E_list.append(E)
        W_list.append(W)
        L_list.append(lam)

    # Fit power-laws for the 3 OddBall relations
    C_NE, a_NE = fit_powerlaw(N_list, E_list)      # E vs N
    C_WE, a_WE = fit_powerlaw(E_list, W_list)      # W vs E
    C_LW, a_LW = fit_powerlaw(W_list, L_list)      # λ1 vs W

    scores = {}
    details = {}

    # Second pass: score each node
    for v, (N, E, W, lam) in feats.items():

        # Predict values from power-laws
        E_pred  = C_NE * (N ** a_NE)
        W_pred  = C_WE * (E ** a_WE)
        L_pred  = C_LW * (W ** a_LW)

        # Compute OddBall anomaly scores
        s1 = oddball_score(E, E_pred)   # CliqueStar
        s2 = oddball_score(W, W_pred)   # HeavyVicinity
        s3 = oddball_score(lam, L_pred) # DominantPair

        combined = s1 + s2 + s3

        scores[v] = combined
        details[v] = {
            "cliqueStar": s1,
            "heavyVicinity": s2,
            "dominantPair": s3
        }

    return scores, details


# --------------------------------------------------
# Main method
# --------------------------------------------------

if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to weighted edge list (u v w)")
    args = parser.parse_args()

    print("Loading graph.")
    G = nx.read_weighted_edgelist(args.input)

    print("Running OddBall Algorithm.")
    scores, details = oddball(G)

    print("\nTop 20 anomalies:")
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
    for node, score in top:
        print(f"{node}\t{score:.4f}")
