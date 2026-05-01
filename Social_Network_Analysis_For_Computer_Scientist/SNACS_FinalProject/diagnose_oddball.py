"""
诊断脚本：检查 Oddball 实现的潜在问题
"""
import numpy as np
import networkx as nx

# 模拟一个简单的测试案例
def test_egonet_issue():
    """
    测试问题1: egonet 定义是否正确
    """
    print("=" * 60)
    print("问题1: Egonet 定义错误")
    print("=" * 60)
    
    # 创建一个简单的图
    G = nx.Graph()
    G.add_edges_from([
        (0, 1, {'weight': 1}),
        (0, 2, {'weight': 1}),
        (1, 2, {'weight': 1}),  # 这条边在邻居之间
        (1, 3, {'weight': 1})
    ])
    
    print(f"原图边: {list(G.edges())}")
    
    # 当前实现
    def get_egonet_current(G, v):
        neighbors = list(G.neighbors(v))
        nodes = neighbors + [v]
        return G.subgraph(nodes)
    
    # 正确实现
    def get_egonet_correct(G, v):
        neighbors = set(G.neighbors(v))
        nodes = neighbors | {v}  # 包含中心节点和所有邻居
        return G.subgraph(nodes)
    
    ego_current = get_egonet_current(G, 0)
    ego_correct = get_egonet_correct(G, 0)
    
    print(f"\n节点0的邻居: {list(G.neighbors(0))}")
    print(f"当前实现 - egonet节点: {list(ego_current.nodes())}")
    print(f"当前实现 - egonet边数: {ego_current.number_of_edges()}")
    print(f"当前实现 - egonet边: {list(ego_current.edges())}")
    
    print(f"\n正确实现 - egonet节点: {list(ego_correct.nodes())}")
    print(f"正确实现 - egonet边数: {ego_correct.number_of_edges()}")
    print(f"正确实现 - egonet边: {list(ego_correct.edges())}")
    
    print(f"\n✓ 两种实现结果相同" if ego_current.number_of_edges() == ego_correct.number_of_edges() else "✗ 实现有差异")
    
    # 测试节点1（它有邻居之间的边）
    print(f"\n节点1的情况:")
    ego_current_1 = get_egonet_current(G, 1)
    ego_correct_1 = get_egonet_correct(G, 1)
    print(f"当前实现 - egonet边: {list(ego_current_1.edges())}")
    print(f"正确实现 - egonet边: {list(ego_correct_1.edges())}")


def test_scoring_formula():
    """
    测试问题2: 评分公式是否正确
    """
    print("\n" + "=" * 60)
    print("问题2: 评分公式检查")
    print("=" * 60)
    
    # 当前实现
    def oddball_score_current(y_true, y_pred):
        if y_true <= 0 or y_pred <= 0:
            return 0.0
        ratio = max(y_true, y_pred) / min(y_true, y_pred)
        diff = np.log(abs(y_true - y_pred) + 1.0)
        return ratio * diff
    
    # 原始OddBall论文的公式
    def oddball_score_original(y_true, y_pred):
        if y_true <= 0 or y_pred <= 0:
            return 0.0
        # O(i) = |y - ŷ| / min(y, ŷ)
        return abs(y_true - y_pred) / min(y_true, y_pred)
    
    # 测试几个案例
    test_cases = [
        (10, 5, "较大偏差"),
        (10, 9, "小偏差"),
        (10, 10, "完全匹配"),
        (5, 10, "预测过高"),
        (100, 10, "极大偏差"),
    ]
    
    print(f"\n{'实际值':<10} {'预测值':<10} {'当前公式':<15} {'原始公式':<15} {'场景':<15}")
    print("-" * 70)
    for y_true, y_pred, desc in test_cases:
        score_current = oddball_score_current(y_true, y_pred)
        score_original = oddball_score_original(y_true, y_pred)
        print(f"{y_true:<10} {y_pred:<10} {score_current:<15.4f} {score_original:<15.4f} {desc:<15}")
    
    print("\n观察：当前公式使用了 ratio * log(diff)，这会放大大偏差的重要性")
    print("原始公式使用 |y - ŷ| / min(y, ŷ)，更直接地衡量相对偏差")


def test_power_law_robustness():
    """
    测试问题3: Power-law 拟合的鲁棒性
    """
    print("\n" + "=" * 60)
    print("问题3: Power-law 拟合鲁棒性")
    print("=" * 60)
    
    # 模拟一些带噪声的数据
    np.random.seed(42)
    X = np.arange(1, 101)
    Y_clean = 2 * X ** 1.5  # 理想的 power-law
    Y_noisy = Y_clean + np.random.normal(0, Y_clean * 0.5)  # 加50%噪声
    
    def fit_powerlaw_basic(X, Y):
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
    
    C_clean, a_clean = fit_powerlaw_basic(X, Y_clean)
    C_noisy, a_noisy = fit_powerlaw_basic(X, Y_noisy)
    
    print(f"理想数据: C={C_clean:.4f}, a={a_clean:.4f} (应该接近 C=2, a=1.5)")
    print(f"噪声数据: C={C_noisy:.4f}, a={a_noisy:.4f}")
    print(f"参数偏差: ΔC={(C_noisy-2)/2*100:.1f}%, Δa={(a_noisy-1.5)/1.5*100:.1f}%")
    
    # 检查异常值的影响
    X_outlier = np.append(X, [1, 2, 3])
    Y_outlier = np.append(Y_clean, [1000, 2000, 3000])  # 添加异常值
    
    C_outlier, a_outlier = fit_powerlaw_basic(X_outlier, Y_outlier)
    print(f"\n含异常值: C={C_outlier:.4f}, a={a_outlier:.4f}")
    print("观察：异常值会显著影响拟合结果，导致预测不准确")


def test_zero_handling():
    """
    测试问题4: 零值和小值的处理
    """
    print("\n" + "=" * 60)
    print("问题4: 零值和边界情况处理")
    print("=" * 60)
    
    def oddball_score(y_true, y_pred):
        if y_true <= 0 or y_pred <= 0:
            return 0.0
        ratio = max(y_true, y_pred) / min(y_true, y_pred)
        diff = np.log(abs(y_true - y_pred) + 1.0)
        return ratio * diff
    
    # 测试边界情况
    test_cases = [
        (0, 5, "实际为0"),
        (5, 0, "预测为0"),
        (0.1, 10, "实际很小"),
        (10, 0.1, "预测很小"),
        (1, 1, "完全相同"),
    ]
    
    print(f"\n{'实际值':<10} {'预测值':<10} {'得分':<15} {'说明':<20}")
    print("-" * 60)
    for y_true, y_pred, desc in test_cases:
        score = oddball_score(y_true, y_pred)
        print(f"{y_true:<10} {y_pred:<10} {score:<15.4f} {desc:<20}")
    
    print("\n问题：当节点孤立或边很少时，E、W、λ1可能为0")
    print("这会导致该节点的异常分数为0，即使它可能很异常")


def analyze_yelp_characteristics():
    """
    测试问题5: Yelp图的特性分析
    """
    print("\n" + "=" * 60)
    print("问题5: Yelp欺诈检测的特殊性")
    print("=" * 60)
    
    print("""
Yelp欺诈节点的典型特征:
1. 共谋团伙：多个账号在相同时间给相同星级（高度同步）
2. 刷单账号：频繁评论、评分模式单一
3. 正常用户：评论时间分散、星级多样化

OddBall原本设计用于检测:
- CliqueStar: 星型结构 vs 团结构
- HeavyVicinity: 边权重异常高
- DominantPair: 特征值异常

问题所在:
✗ Yelp的欺诈模式不一定表现为结构异常
✗ RUR(同用户)+RTR∩RSR可能无法充分捕捉欺诈信号
✗ 正常活跃用户可能因为高度数而被误判为异常

建议的改进方向:
1. 使用更合适的图构建策略（如R-U-R-U bipartite）
2. 添加时序特征和星级分布特征
3. 考虑使用有监督方法或图神经网络
4. 调整评分公式以适应欺诈检测场景
    """)


if __name__ == "__main__":
    test_egonet_issue()
    test_scoring_formula()
    test_power_law_robustness()
    test_zero_handling()
    analyze_yelp_characteristics()
    
    print("\n" + "=" * 60)
    print("总结与建议")
    print("=" * 60)
    print("""
发现的主要问题:
1. ✓ Egonet定义可能正确但需要确认是否包含邻居间的边
2. ? 评分公式与原始OddBall论文可能不同
3. ✗ Power-law拟合对噪声和异常值敏感
4. ✗ 零值处理可能导致异常节点得分为0
5. ✗ Yelp图的构建策略可能不适合OddBall方法

建议的修复优先级:
[高] 检查并修复评分公式（参考原始论文）
[高] 改进图构建策略，更好地捕捉欺诈模式
[中] 添加鲁棒的power-law拟合（如RANSAC）
[中] 改进零值和小值的处理
[低] 优化egonet计算效率
    """)
