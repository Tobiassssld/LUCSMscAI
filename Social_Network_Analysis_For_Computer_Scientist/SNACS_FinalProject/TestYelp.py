import scipy.io
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import Oddball  # 导入您的 Oddball 模块


def load_and_convert_yelpchi(filepath):
    print(f"Loading {filepath}...")
    data = scipy.io.loadmat(filepath)

    net_rur = data['net_rur']  # 同用户
    net_rtr = data['net_rtr']  # 同时间
    net_rsr = data['net_rsr']  # 同星级
    labels = data['label'].flatten()

    print("Constructing weighted graph with COLLUSION strategy...")

    # 1. 基础骨架：同用户 (RUR)
    # 这是必须的，保留用户的历史行为特征
    adj_rur = net_rur.astype(float)

    # 2. 共谋信号：同时间 AND 同星级 (RTR * RSR)
    # 只有当两个评论既是同一时间发的，又是同一星级时，才认为是“共谋边”
    # multiply 是点乘 (Intersection)，可以极大地过滤掉噪声
    adj_collusion = net_rtr.multiply(net_rsr)

    # 3. 合并
    # 权重逻辑：
    # - 仅同用户: weight = 1 (正常)
    # - 不同用户但共谋: weight = 1 (可疑连接)
    # - 同用户且重复发同样星级: weight = 2 (非常可疑的刷单狂)
    weighted_adj = adj_rur + adj_collusion

    G = nx.from_scipy_sparse_array(weighted_adj, create_using=nx.Graph, edge_attribute='weight')

    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print("Graph logic: RUR + (RTR & RSR). Captures users and spam gangs.")

    return G, labels

def evaluate_oddball(scores, labels):
    # 对齐分数和标签
    # scores 是字典 {node_id: score}，我们需要将其转换为列表以匹配 labels
    y_true = []
    y_scores = []

    # YelpChi 的节点 ID 是从 0 到 N-1 的索引
    for node_id in range(len(labels)):
        y_true.append(labels[node_id])
        # 如果某个节点是孤立点，可能没有被 Oddball 评分，默认给 0 分
        y_scores.append(scores.get(node_id, 0.0))

    # 计算 AUC-ROC 和 AP (Average Precision)
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    print("-" * 30)
    print(f"Evaluation Results:")
    print(f"ROC-AUC Score: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    # 1. 加载数据
    try:
        G, labels = load_and_convert_yelpchi('YelpChi.mat')
    except FileNotFoundError:
        print("Error: YelpChi.mat not found. Please download it first.")
        exit()

    # 2. 运行您的 Oddball 算法
    print("Running Oddball algorithm (this may take a while)...")
    scores, details = Oddball.oddball(G)

    # 3. 评估结果
    evaluate_oddball(scores, labels)

    # 4. (可选) 查看 Top 异常节点是否确实是标签 1
    print("\nTop 10 Anomalies found by Oddball:")
    top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    for node, score in top_nodes:
        is_fraud = "FRAUD" if labels[node] == 1 else "Normal"
        print(f"Node {node}: Score {score:.4f} ({is_fraud})")