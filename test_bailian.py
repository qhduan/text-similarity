"""
百炼API文本相似度测试脚本
"""

import os
import asyncio
from text_similarity import TextSimilarityBailian


def test_sync():
    """同步接口测试"""
    print("=" * 60)
    print("测试 TextSimilarityBailian (同步接口)")
    print("=" * 60)
    
    # 检查环境变量
    if not os.environ.get("BAILIAN_TOKEN"):
        print("\n错误: 未设置环境变量 BAILIAN_TOKEN")
        print("请设置: export BAILIAN_TOKEN='your-api-token'")
        return
    
    # 初始化
    print("\n1. 初始化 TextSimilarityBailian...")
    ts = TextSimilarityBailian()
    print("   ✓ 初始化成功")
    print(f"   模型: {ts.model}")
    print(f"   向量维度: {ts.dimensions}")
    
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
    print(f"   第一个向量的前5个值: {vectors[0][:5]}")
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
        print(f"     - {text}: {score:.4f} (索引: {idx})")
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
    print(f"   保留的文本:")
    for idx in kept:
        print(f"     [{idx}] {dup_texts[idx]}")
    print("   ✓ 文本去重成功")
    
    print("\n" + "=" * 60)
    print("所有同步测试通过！✓")
    print("=" * 60)


async def test_async():
    """异步接口测试"""
    print("\n" + "=" * 60)
    print("测试 TextSimilarityBailian (异步接口)")
    print("=" * 60)
    
    # 检查环境变量
    if not os.environ.get("BAILIAN_TOKEN"):
        print("\n错误: 未设置环境变量 BAILIAN_TOKEN")
        return
    
    # 初始化
    print("\n1. 初始化 TextSimilarityBailian...")
    ts = TextSimilarityBailian()
    print("   ✓ 初始化成功")
    
    # 测试异步相似度计算
    print("\n2. 测试异步相似度计算...")
    text1 = "你好世界"
    text2 = "你好中国"
    similarity = await ts.calculate_similarity_async(text1, text2)
    print(f"   文本1: {text1}")
    print(f"   文本2: {text2}")
    print(f"   相似度: {similarity:.4f}")
    print("   ✓ 异步相似度计算成功")
    
    # 测试异步向量化
    print("\n3. 测试异步向量化...")
    texts = ["自然语言处理", "计算机视觉", "语音识别"]
    vectors = await ts.get_vectors_async(texts)
    print(f"   文本数量: {len(texts)}")
    print(f"   向量形状: {vectors.shape}")
    print("   ✓ 异步向量化成功")
    
    # 测试异步相似度矩阵
    print("\n4. 测试异步相似度矩阵...")
    matrix = await ts.get_similarity_matrix_async(texts)
    print(f"   矩阵形状: {matrix.shape}")
    print("   ✓ 异步相似度矩阵计算成功")
    
    # 测试异步查找最相似文本
    print("\n5. 测试异步查找最相似文本...")
    query = "深度神经网络"
    candidates = ["机器学习", "强化学习", "迁移学习", "联邦学习"]
    results = await ts.find_most_similar_async(query, candidates, top_k=3)
    print(f"   查询: {query}")
    print(f"   最相似的3个结果:")
    for rank, (text, score, idx) in enumerate(results, 1):
        print(f"     {rank}. {text}: {score:.4f} (索引: {idx})")
    print("   ✓ 异步查找最相似文本成功")
    
    # 测试异步文本去重
    print("\n6. 测试异步文本去重...")
    dup_texts = [
        "Python编程语言",
        "Java开发",
        "Python程序设计",  # 与第1个相似
        "数据库设计",
        "Java编程",  # 与第2个相似
        "前端开发",
    ]
    kept = await ts.deduplicate_texts_async(dup_texts, keep_count=4, similarity_threshold=0.80)
    print(f"   原始文本数: {len(dup_texts)}")
    print(f"   保留的索引: {kept}")
    print("   ✓ 异步文本去重成功")
    
    print("\n" + "=" * 60)
    print("所有异步测试通过！✓")
    print("=" * 60)


async def test_concurrent():
    """并发性能测试"""
    print("\n" + "=" * 60)
    print("测试并发性能")
    print("=" * 60)
    
    if not os.environ.get("BAILIAN_TOKEN"):
        print("\n错误: 未设置环境变量 BAILIAN_TOKEN")
        return
    
    ts = TextSimilarityBailian()
    
    # 准备测试数据
    texts = [
        "机器学习算法研究",
        "深度学习模型训练",
        "自然语言处理技术",
        "计算机视觉应用",
        "数据挖掘分析",
    ]
    
    print(f"\n并发计算 {len(texts)} 个文本的相似度矩阵...")
    import time
    start_time = time.time()
    
    matrix = await ts.get_similarity_matrix_async(texts)
    
    elapsed = time.time() - start_time
    print(f"   耗时: {elapsed:.2f}秒")
    print(f"   矩阵形状: {matrix.shape}")
    print("   ✓ 并发测试完成")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("百炼API文本相似度包 - 测试脚本")
    print("=" * 60)
    
    try:
        # 同步测试
        test_sync()
        
        # 异步测试
        asyncio.run(test_async())
        
        # 并发测试
        asyncio.run(test_concurrent())
        
        print("\n" + "=" * 60)
        print("所有测试完成！✓")
        print("=" * 60)
        
    except ValueError as e:
        print(f"\n配置错误: {e}")
        print("\n使用说明:")
        print("  1. 设置环境变量: export BAILIAN_TOKEN='your-api-token'")
        print("  2. 运行测试: python test_bailian.py")
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n请确保:")
        print("  1. 已安装依赖: pip install -r requirements.txt")
        print("  2. 已设置 BAILIAN_TOKEN 环境变量")
        print("  3. API Token 有效且有足够的配额")


if __name__ == "__main__":
    main()
