"""
简单测试脚本 - 验证text_similarity包是否正常工作
"""

from text_similarity import TextSimilarity


def test_basic():
    """基础功能测试"""
    print("=" * 60)
    print("测试 text_similarity 包")
    print("=" * 60)
    
    # 初始化
    print("\n1. 初始化 TextSimilarity...")
    ts = TextSimilarity()
    print("   ✓ 初始化成功")
    
    # 测试相似度计算
    print("\n2. 测试两个字符串的相似度计算...")
    text1 = "机器学习"
    text2 = "深度学习"
    similarity = ts.calculate_similarity(text1, text2)
    print(f"   文本1: {text1}")
    print(f"   文本2: {text2}")
    print(f"   相似度: {similarity:.4f}")
    print("   ✓ 相似度计算成功")
    
    # 测试向量化
    print("\n3. 测试文本向量化...")
    texts = ["机器学习", "深度学习", "人工智能"]
    vectors = ts.get_vectors(texts)
    print(f"   文本数量: {len(texts)}")
    print(f"   向量形状: {vectors.shape}")
    print("   ✓ 向量化成功")
    
    # 测试相似度矩阵
    print("\n4. 测试相似度矩阵...")
    matrix = ts.get_similarity_matrix(texts)
    print(f"   矩阵形状: {matrix.shape}")
    print(f"   矩阵内容:\n{matrix}")
    print("   ✓ 相似度矩阵计算成功")
    
    # 测试查找最相似文本
    print("\n5. 测试查找最相似文本...")
    query = "机器学习算法"
    candidates = ["深度学习", "云计算", "人工智能", "数据库"]
    results = ts.find_most_similar(query, candidates, top_k=2)
    print(f"   查询: {query}")
    print(f"   最相似的2个结果:")
    for text, score, idx in results:
        print(f"     - {text}: {score:.4f}")
    print("   ✓ 查找最相似文本成功")
    
    # 测试文本去重
    print("\n6. 测试文本去重...")
    dup_texts = [
        "机器学习是AI的分支",
        "深度学习很重要",
        "机器学习属于人工智能",  # 与第1个相似
        "自然语言处理",
        "深度学习是重要技术",  # 与第2个相似
    ]
    kept = ts.deduplicate_texts(dup_texts, keep_count=3, similarity_threshold=0.85)
    print(f"   原始文本数: {len(dup_texts)}")
    print(f"   保留文本数: {len(kept)}")
    print(f"   保留的索引: {kept}")
    print("   ✓ 文本去重成功")
    
    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_basic()
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n请确保已安装依赖:")
        print("  pip install -r requirements.txt")
