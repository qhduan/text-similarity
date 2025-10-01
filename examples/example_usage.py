"""
文本相似度包使用示例 - 基于SentenceTransformers
"""

from text_similarity import TextSimilarity


def example_basic_similarity():
    """示例1: 基础相似度计算"""
    print("=" * 70)
    print("示例1: 基础相似度计算")
    print("=" * 70)
    
    ts = TextSimilarity()
    
    # 中文相似度
    text1 = "机器学习是人工智能的一个分支"
    text2 = "深度学习是机器学习的一个分支"
    
    print(f"\n文本1: {text1}")
    print(f"文本2: {text2}")
    
    similarity = ts.calculate_similarity(text1, text2)
    print(f"相似度: {similarity:.4f}")
    
    # 英文相似度
    text3 = "The weather is nice today"
    text4 = "Today's weather is very good"
    
    print(f"\n文本1: {text3}")
    print(f"文本2: {text4}")
    
    similarity = ts.calculate_similarity(text3, text4)
    print(f"相似度: {similarity:.4f}")
    
    # 跨语言相似度
    text5 = "你好世界"
    text6 = "Hello world"
    
    print(f"\n文本1: {text5}")
    print(f"文本2: {text6}")
    
    similarity = ts.calculate_similarity(text5, text6)
    print(f"跨语言相似度: {similarity:.4f}")


def example_get_vectors():
    """示例2: 获取文本向量"""
    print("\n" + "=" * 70)
    print("示例2: 获取文本向量")
    print("=" * 70)
    
    ts = TextSimilarity()
    
    texts = [
        "机器学习",
        "深度学习",
        "人工智能",
        "自然语言处理",
        "计算机视觉"
    ]
    
    print(f"\n文本列表: {texts}")
    
    vectors = ts.get_vectors(texts)
    print(f"\n向量形状: {vectors.shape}")
    print(f"每个文本的向量维度: {vectors[0].shape[0]}")
    print(f"\n第一个文本 '{texts[0]}' 的向量前10个值:")
    print(vectors[0][:10])


def example_similarity_matrix():
    """示例3: 计算相似度矩阵"""
    print("\n" + "=" * 70)
    print("示例3: 计算相似度矩阵")
    print("=" * 70)
    
    ts = TextSimilarity()
    
    texts = [
        "机器学习",
        "深度学习",
        "人工智能",
        "数据科学"
    ]
    
    print(f"\n文本列表: {texts}")
    
    similarity_matrix = ts.get_similarity_matrix(texts)
    print(f"\n相似度矩阵形状: {similarity_matrix.shape}")
    print("\n相似度矩阵:")
    print(similarity_matrix)
    
    # 打印格式化的相似度矩阵
    print("\n格式化的相似度矩阵:")
    print("         ", end="")
    for text in texts:
        print(f"{text:>8}", end=" ")
    print()
    
    for i, text1 in enumerate(texts):
        print(f"{text1:>8}", end=" ")
        for j, text2 in enumerate(texts):
            print(f"{similarity_matrix[i][j]:>8.4f}", end=" ")
        print()


def example_find_most_similar():
    """示例4: 查找最相似的文本"""
    print("\n" + "=" * 70)
    print("示例4: 查找最相似的文本")
    print("=" * 70)
    
    ts = TextSimilarity()
    
    query = "机器学习算法"
    candidates = [
        "深度学习神经网络",
        "云计算技术架构",
        "人工智能应用",
        "数据库管理系统",
        "自然语言处理",
        "计算机视觉识别",
        "大数据分析",
        "区块链技术"
    ]
    
    print(f"\n查询文本: {query}")
    print(f"\n候选文本列表:")
    for i, text in enumerate(candidates):
        print(f"  {i}: {text}")
    
    # 找出最相似的前5个
    results = ts.find_most_similar(query, candidates, top_k=5)
    
    print(f"\n与 '{query}' 最相似的前5个文本:")
    for rank, (text, score, idx) in enumerate(results, 1):
        print(f"  {rank}. {text} (相似度: {score:.4f}, 索引: {idx})")


def example_multilingual():
    """示例5: 多语言和跨语言相似度"""
    print("\n" + "=" * 70)
    print("示例5: 多语言和跨语言相似度")
    print("=" * 70)
    
    ts = TextSimilarity()
    
    # 多语言文本
    texts = [
        "机器学习",           # 中文
        "Machine Learning",   # 英文
        "Apprentissage automatique",  # 法语
        "深度学习",           # 中文
        "Deep Learning",      # 英文
    ]
    
    print("\n多语言文本列表:")
    for i, text in enumerate(texts):
        print(f"  {i}: {text}")
    
    similarity_matrix = ts.get_similarity_matrix(texts)
    
    print("\n跨语言相似度示例:")
    print(f"  '{texts[0]}' vs '{texts[1]}': {similarity_matrix[0][1]:.4f}")
    print(f"  '{texts[0]}' vs '{texts[2]}': {similarity_matrix[0][2]:.4f}")
    print(f"  '{texts[3]}' vs '{texts[4]}': {similarity_matrix[3][4]:.4f}")


def example_deduplicate_texts():
    """示例6: 文本去重"""
    print("\n" + "=" * 70)
    print("示例6: 文本去重")
    print("=" * 70)
    
    ts = TextSimilarity()
    
    # 包含重复文本的列表
    texts = [
        "机器学习是人工智能的分支",
        "深度学习是机器学习的方法",
        "机器学习属于人工智能领域",  # 与第1个相似
        "自然语言处理很重要",
        "NLP是自然语言处理的简称",  # 与第4个相似
        "计算机视觉是AI的应用",
        "深度学习使用神经网络",  # 与第2个相似
        "数据科学需要统计知识",
    ]
    
    print(f"\n原始文本列表 (共{len(texts)}个):")
    for i, text in enumerate(texts):
        print(f"  [{i}] {text}")
    
    # 从8个文本中保留5个不重复的
    kept_indices = ts.deduplicate_texts(texts, keep_count=5, similarity_threshold=0.85)
    
    print(f"\n保留的文本索引: {kept_indices}")
    print(f"\n保留的文本 (共{len(kept_indices)}个):")
    for idx in kept_indices:
        print(f"  [{idx}] {texts[idx]}")
    
    # 显示被移除的文本
    removed_indices = [i for i in range(len(texts)) if i not in kept_indices]
    print(f"\n被移除的文本 (共{len(removed_indices)}个):")
    for idx in removed_indices:
        print(f"  [{idx}] {texts[idx]}")


def example_different_models():
    """示例7: 使用不同的模型"""
    print("\n" + "=" * 70)
    print("示例7: 使用不同的模型")
    print("=" * 70)
    
    text1 = "人工智能正在改变世界"
    text2 = "AI is changing the world"
    
    print(f"\n文本1: {text1}")
    print(f"文本2: {text2}")
    
    # 默认模型 (Qwen3-Embedding-0.6B)
    print("\n1. 使用默认模型 (Qwen/Qwen3-Embedding-0.6B):")
    ts1 = TextSimilarity()
    sim1 = ts1.calculate_similarity(text1, text2)
    print(f"   相似度: {sim1:.4f}")
    
    # 其他模型（注释掉以避免下载大模型，可根据需要启用）
    # print("\n2. 使用多语言模型 (paraphrase-multilingual-mpnet-base-v2):")
    # ts2 = TextSimilarity(model_name='paraphrase-multilingual-mpnet-base-v2')
    # sim2 = ts2.calculate_similarity(text1, text2)
    # print(f"   相似度: {sim2:.4f}")


def main():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("文本相似度包 - 使用示例")
    print("基于 SentenceTransformers 多语言模型")
    print("=" * 70)
    
    try:
        example_basic_similarity()
        example_get_vectors()
        example_similarity_matrix()
        example_find_most_similar()
        example_multilingual()
        example_deduplicate_texts()
        example_different_models()
        
        print("\n" + "=" * 70)
        print("所有示例运行完成！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n提示: 确保已安装所有依赖:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
