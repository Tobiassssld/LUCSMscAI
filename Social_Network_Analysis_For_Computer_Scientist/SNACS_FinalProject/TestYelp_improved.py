import scipy.io
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import Oddball_improved as Oddball


def load_and_convert_yelpchi(filepath, strategy="collusion"):
    """
    Load YelpChi dataset and construct graph with different strategies.
    
    Strategies:
    - "collusion": RUR + (RTR ∩ RSR) - focuses on spam gangs
    - "rur_only": Only same user edges
    - "temporal": RUR + RTR - includes temporal patterns
    - "rating": RUR + RSR - includes rating patterns
    - "all": RUR + RTR + RSR - all signals
    """
    print(f"Loading {filepath}...")
    data = scipy.io.loadmat(filepath)

    net_rur = data['net_rur']  # Same user
    net_rtr = data['net_rtr']  # Same time
    net_rsr = data['net_rsr']  # Same rating
    labels = data['label'].flatten()

    print(f"Constructing weighted graph with '{strategy}' strategy...")

    if strategy == "collusion":
        # Original: RUR + (RTR ∩ RSR)
        adj_collusion = net_rtr.multiply(net_rsr)
        weighted_adj = net_rur.astype(float) + adj_collusion
        desc = "RUR + (RTR & RSR). Captures users and spam gangs."
        
    elif strategy == "rur_only":
        # Only same-user edges
        weighted_adj = net_rur.astype(float)
        desc = "RUR only. Only same-user connections."
        
    elif strategy == "temporal":
        # RUR + temporal signal
        weighted_adj = net_rur.astype(float) + net_rtr.astype(float)
        desc = "RUR + RTR. Includes temporal patterns."
        
    elif strategy == "rating":
        # RUR + rating signal
        weighted_adj = net_rur.astype(float) + net_rsr.astype(float)
        desc = "RUR + RSR. Includes rating patterns."
        
    elif strategy == "all":
        # All signals
        weighted_adj = net_rur.astype(float) + net_rtr.astype(float) + net_rsr.astype(float)
        desc = "RUR + RTR + RSR. All signals combined."
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    G = nx.from_scipy_sparse_array(weighted_adj, create_using=nx.Graph, edge_attribute='weight')

    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print(f"Graph logic: {desc}")

    return G, labels


def evaluate_oddball(scores, labels, verbose=True):
    """Evaluate OddBall scores against ground truth labels"""
    y_true = []
    y_scores = []

    for node_id in range(len(labels)):
        y_true.append(labels[node_id])
        y_scores.append(scores.get(node_id, 0.0))

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Calculate metrics
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    if verbose:
        print("-" * 60)
        print(f"Evaluation Results:")
        print(f"  ROC-AUC Score: {auc:.4f}")
        print(f"  Average Precision: {ap:.4f}")
        
        # Additional statistics
        fraud_indices = y_true == 1
        normal_indices = y_true == 0
        
        fraud_scores = y_scores[fraud_indices]
        normal_scores = y_scores[normal_indices]
        
        print(f"\nScore Statistics:")
        print(f"  Fraud nodes ({fraud_indices.sum()}):")
        print(f"    Mean: {fraud_scores.mean():.4f}, Median: {np.median(fraud_scores):.4f}")
        print(f"  Normal nodes ({normal_indices.sum()}):")
        print(f"    Mean: {normal_scores.mean():.4f}, Median: {np.median(normal_scores):.4f}")
        print(f"  Score separation: {fraud_scores.mean() - normal_scores.mean():.4f}")
        print("-" * 60)

    return auc, ap


def compare_configurations(G, labels):
    """Compare different OddBall configurations"""
    
    print("\n" + "="*60)
    print("COMPARING DIFFERENT ODDBALL CONFIGURATIONS")
    print("="*60)
    
    configs = [
        ("Original Formula", True, False),
        ("Modified Formula", False, False),
        ("Original + Normalize", True, True),
        ("Modified + Normalize", False, True),
    ]
    
    results = []
    
    for name, use_original, normalize in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        
        scores, details = Oddball.oddball(
            G, 
            use_original_formula=use_original,
            normalize=normalize,
            verbose=False
        )
        
        auc, ap = evaluate_oddball(scores, labels, verbose=True)
        results.append((name, auc, ap, scores, details))
        
        # Show top 10
        print(f"\nTop 10 Anomalies:")
        top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        for node, score in top_nodes:
            is_fraud = "FRAUD" if labels[node] == 1 else "Normal"
            print(f"  Node {node}: Score {score:.4f} ({is_fraud})")
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Configuration':<25} {'AUC':<10} {'AP':<10}")
    print("-"*60)
    for name, auc, ap, _, _ in results:
        print(f"{name:<25} {auc:<10.4f} {ap:<10.4f}")
    
    # Find best
    best_idx = max(range(len(results)), key=lambda i: results[i][1])
    print(f"\nBest configuration: {results[best_idx][0]} (AUC={results[best_idx][1]:.4f})")
    
    return results


def analyze_fraud_patterns(G, labels, scores, details):
    """Analyze what patterns fraud nodes exhibit"""
    
    print("\n" + "="*60)
    print("FRAUD PATTERN ANALYSIS")
    print("="*60)
    
    fraud_nodes = [i for i, label in enumerate(labels) if label == 1]
    normal_nodes = [i for i, label in enumerate(labels) if label == 0]
    
    # Sample some nodes for detailed analysis
    np.random.seed(42)
    sample_fraud = np.random.choice(fraud_nodes, min(100, len(fraud_nodes)), replace=False)
    sample_normal = np.random.choice(normal_nodes, min(100, len(normal_nodes)), replace=False)
    
    # Collect features
    def get_features(nodes):
        features = {
            'N': [], 'E': [], 'W': [], 'lambda1': [],
            'cliqueStar': [], 'heavyVicinity': [], 'dominantPair': [],
            'total_score': []
        }
        for node in nodes:
            if node in details:
                d = details[node]
                features['N'].append(d['N'])
                features['E'].append(d['E'])
                features['W'].append(d['W'])
                features['lambda1'].append(d['lambda1'])
                features['cliqueStar'].append(d['cliqueStar'])
                features['heavyVicinity'].append(d['heavyVicinity'])
                features['dominantPair'].append(d['dominantPair'])
                features['total_score'].append(scores[node])
        return {k: np.array(v) for k, v in features.items()}
    
    fraud_feats = get_features(sample_fraud)
    normal_feats = get_features(sample_normal)
    
    print(f"\nComparing {len(sample_fraud)} fraud vs {len(sample_normal)} normal nodes:")
    print(f"\n{'Feature':<20} {'Fraud Mean':<15} {'Normal Mean':<15} {'Ratio':<10}")
    print("-"*65)
    
    for feat in ['N', 'E', 'W', 'lambda1']:
        fraud_mean = fraud_feats[feat].mean()
        normal_mean = normal_feats[feat].mean()
        ratio = fraud_mean / (normal_mean + 1e-10)
        print(f"{feat:<20} {fraud_mean:<15.2f} {normal_mean:<15.2f} {ratio:<10.2f}")
    
    print(f"\n{'Score Component':<20} {'Fraud Mean':<15} {'Normal Mean':<15} {'Ratio':<10}")
    print("-"*65)
    
    for feat in ['cliqueStar', 'heavyVicinity', 'dominantPair', 'total_score']:
        fraud_mean = fraud_feats[feat].mean()
        normal_mean = normal_feats[feat].mean()
        ratio = fraud_mean / (normal_mean + 1e-10)
        print(f"{feat:<20} {fraud_mean:<15.4f} {normal_mean:<15.4f} {ratio:<10.2f}")


def test_graph_strategies(filepath):
    """Test different graph construction strategies"""
    
    print("\n" + "="*80)
    print("TESTING DIFFERENT GRAPH CONSTRUCTION STRATEGIES")
    print("="*80)
    
    strategies = ["rur_only", "temporal", "rating", "collusion", "all"]
    
    strategy_results = []
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'='*80}")
        
        G, labels = load_and_convert_yelpchi(filepath, strategy=strategy)
        
        # Run with original formula
        scores, details = Oddball.oddball(G, use_original_formula=True, verbose=False)
        auc, ap = evaluate_oddball(scores, labels, verbose=True)
        
        strategy_results.append((strategy, auc, ap))
    
    # Summary
    print("\n" + "="*80)
    print("GRAPH STRATEGY COMPARISON")
    print("="*80)
    print(f"{'Strategy':<20} {'AUC':<10} {'AP':<10}")
    print("-"*50)
    for strategy, auc, ap in strategy_results:
        print(f"{strategy:<20} {auc:<10.4f} {ap:<10.4f}")
    
    best_idx = max(range(len(strategy_results)), key=lambda i: strategy_results[i][1])
    print(f"\nBest strategy: {strategy_results[best_idx][0]} (AUC={strategy_results[best_idx][1]:.4f})")


if __name__ == "__main__":
    filepath = 'YelpChi.mat'
    
    try:
        # Test 1: Compare different graph construction strategies
        print("TEST 1: Graph Construction Strategies")
        test_graph_strategies(filepath)
        
        # Test 2: Compare OddBall configurations on best strategy
        print("\n\nTEST 2: OddBall Configuration Comparison")
        print("Using 'collusion' strategy (original approach)")
        G, labels = load_and_convert_yelpchi(filepath, strategy="collusion")
        results = compare_configurations(G, labels)
        
        # Test 3: Analyze fraud patterns
        print("\n\nTEST 3: Fraud Pattern Analysis")
        best_config = max(results, key=lambda x: x[1])
        _, _, _, best_scores, best_details = best_config
        analyze_fraud_patterns(G, labels, best_scores, best_details)
        
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please download it first.")
        print("You can download from: https://github.com/YingtongDou/CARE-GNN")
