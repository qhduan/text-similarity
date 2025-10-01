"""
核心相似度计算模块 - 基于SentenceTransformers
"""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer, util


class TextSimilarity:
    """
    文本相似度计算类 - 使用SentenceTransformers多语言模型
    
    提供基于深度学习的文本相似度计算，支持50+种语言。
    """
    
    def __init__(self, model_name: Optional[str] = None, model_kwargs: Optional[dict] = None, tokenizer_kwargs: Optional[dict] = None):
        """
        初始化TextSimilarity对象
        
        Args:
            model_name (str, optional): SentenceTransformers模型名称
                推荐的多语言模型:
                - 'Qwen/Qwen3-Embedding-0.6B' (默认，高质量中英文嵌入模型)
                - 'paraphrase-multilingual-MiniLM-L12-v2' (支持50+语言，轻量快速)
                - 'paraphrase-multilingual-mpnet-base-v2' (更高精度)
            model_kwargs (dict, optional): 传递给模型的参数
            tokenizer_kwargs (dict, optional): 传递给tokenizer的参数
        
        Examples:
            >>> ts = TextSimilarity()
            >>> ts = TextSimilarity(model_name='paraphrase-multilingual-mpnet-base-v2')
            >>> # 使用flash attention加速
            >>> ts = TextSimilarity(
            ...     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
            ...     tokenizer_kwargs={"padding_side": "left"}
            ... )
        """
        self.model_name = model_name or 'Qwen/Qwen3-Embedding-0.6B'
        
        # 构建模型参数
        init_kwargs = {}
        if model_kwargs:
            init_kwargs['model_kwargs'] = model_kwargs
        if tokenizer_kwargs:
            init_kwargs['tokenizer_kwargs'] = tokenizer_kwargs
        
        self.model = SentenceTransformer(self.model_name, **init_kwargs)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个字符串之间的相似度
        
        Args:
            text1 (str): 第一个字符串
            text2 (str): 第二个字符串
        
        Returns:
            float: 相似度分数，范围为[0, 1]，1表示完全相同
        
        Examples:
            >>> ts = TextSimilarity()
            >>> similarity = ts.calculate_similarity("你好世界", "你好中国")
            >>> print(f"相似度: {similarity:.4f}")
            相似度: 0.7234
            
            >>> # 跨语言相似度
            >>> similarity = ts.calculate_similarity("Hello world", "你好世界")
            >>> print(f"跨语言相似度: {similarity:.4f}")
            跨语言相似度: 0.6521
        """
        # 编码文本为向量
        embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
        
        # 计算余弦相似度
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        
        return float(similarity.item())
    
    def get_vectors(self, texts: List[str]) -> np.ndarray:
        """
        返回N个字符串的向量表示
        
        Args:
            texts (List[str]): 字符串列表
        
        Returns:
            np.ndarray: 形状为(N, M)的向量矩阵
                - N为字符串数量
                - M为特征维度（取决于模型，通常为384或768）
        
        Examples:
            >>> ts = TextSimilarity()
            >>> texts = ["机器学习", "深度学习", "人工智能"]
            >>> vectors = ts.get_vectors(texts)
            >>> print(f"向量形状: {vectors.shape}")
            向量形状: (3, 384)
        """
        if not texts:
            raise ValueError("文本列表不能为空")
        
        # 使用SentenceTransformers编码
        vectors = self.model.encode(texts, convert_to_numpy=True)
        
        return vectors
    
    def get_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        返回N个字符串的相似度矩阵
        
        Args:
            texts (List[str]): 字符串列表
        
        Returns:
            np.ndarray: 形状为(N, N)的相似度矩阵
                - matrix[i][j]表示第i个和第j个字符串的相似度
                - 对角线元素为1.0（自己与自己的相似度）
                - 矩阵是对称的
        
        Examples:
            >>> ts = TextSimilarity()
            >>> texts = ["机器学习", "深度学习", "人工智能", "数据科学"]
            >>> matrix = ts.get_similarity_matrix(texts)
            >>> print(f"相似度矩阵形状: {matrix.shape}")
            相似度矩阵形状: (4, 4)
            >>> print(f"机器学习 vs 深度学习: {matrix[0][1]:.4f}")
            机器学习 vs 深度学习: 0.8523
        """
        if not texts:
            raise ValueError("文本列表不能为空")
        
        # 获取所有文本的向量
        vectors = self.get_vectors(texts)
        
        # 计算相似度矩阵
        similarity_matrix = util.cos_sim(vectors, vectors)
        
        return similarity_matrix.cpu().numpy()
    
    def find_most_similar(self, query: str, candidates: List[str], top_k: int = 5) -> List[tuple]:
        """
        在候选文本中找到与查询文本最相似的top_k个结果
        
        Args:
            query (str): 查询文本
            candidates (List[str]): 候选文本列表
            top_k (int): 返回最相似的前k个结果，默认为5
        
        Returns:
            List[tuple]: 包含(文本, 相似度分数, 索引)的列表，按相似度降序排列
        
        Examples:
            >>> ts = TextSimilarity()
            >>> query = "机器学习算法"
            >>> candidates = ["深度学习", "神经网络", "数据挖掘", "云计算", "人工智能"]
            >>> results = ts.find_most_similar(query, candidates, top_k=3)
            >>> for text, score, idx in results:
            ...     print(f"{text}: {score:.4f}")
            深度学习: 0.8234
            人工智能: 0.7892
            神经网络: 0.7654
        """
        if not candidates:
            raise ValueError("候选文本列表不能为空")
        
        # 编码查询和候选文本
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        candidate_embeddings = self.model.encode(candidates, convert_to_tensor=True)
        
        # 计算相似度
        similarities = util.cos_sim(query_embedding, candidate_embeddings)[0]
        
        # 获取top_k结果
        top_k = min(top_k, len(candidates))
        top_results = similarities.topk(k=top_k)
        
        # 构建结果列表
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append((candidates[idx.item()], float(score.item()), int(idx.item())))
        
        return results
    
    def deduplicate_texts(self, texts: List[str], keep_count: int, similarity_threshold: float = 0.9) -> List[int]:
        """
        文本去重：从N个文本中保留指定数量的文本，返回保留文本的索引
        
        基于相似度阈值进行去重，优先保留出现较早的文本。
        
        Args:
            texts (List[str]): 输入的N个文本列表
            keep_count (int): 要保留的文本数量
            similarity_threshold (float): 相似度阈值，默认0.9。
                当两个文本相似度超过此阈值时，认为它们重复，只保留一个
        
        Returns:
            List[int]: 保留的文本在原列表中的索引列表，按原始顺序排列
        
        Examples:
            >>> ts = TextSimilarity()
            >>> texts = [
            ...     "机器学习是人工智能的分支",
            ...     "深度学习是机器学习的方法",
            ...     "机器学习属于人工智能领域",  # 与第1个相似
            ...     "自然语言处理很重要",
            ...     "NLP是自然语言处理",  # 与第4个相似
            ... ]
            >>> indices = ts.deduplicate_texts(texts, keep_count=3, similarity_threshold=0.85)
            >>> print(f"保留的文本索引: {indices}")
            保留的文本索引: [0, 1, 3]
            >>> print(f"保留的文本:")
            >>> for idx in indices:
            ...     print(f"  [{idx}] {texts[idx]}")
        """
        if not texts:
            raise ValueError("文本列表不能为空")
        
        if keep_count <= 0:
            raise ValueError("keep_count必须大于0")
        
        if keep_count >= len(texts):
            # 如果要保留的数量大于等于文本总数，直接返回所有索引
            return list(range(len(texts)))
        
        # 获取所有文本的向量
        vectors = self.get_vectors(texts)
        
        # 计算相似度矩阵
        similarity_matrix = util.cos_sim(vectors, vectors).cpu().numpy()
        
        # 去重逻辑：使用贪心算法
        kept_indices = []
        available_indices = set(range(len(texts)))
        
        for i in range(len(texts)):
            if i not in available_indices:
                continue
            
            # 保留当前文本
            kept_indices.append(i)
            
            # 如果已经保留了足够的文本，退出
            if len(kept_indices) >= keep_count:
                break
            
            # 移除与当前文本相似的其他文本
            for j in range(i + 1, len(texts)):
                if j in available_indices and similarity_matrix[i][j] >= similarity_threshold:
                    available_indices.remove(j)
        
        # 如果保留的文本数量不足，从剩余文本中补充
        if len(kept_indices) < keep_count:
            remaining = sorted(available_indices)
            needed = keep_count - len(kept_indices)
            kept_indices.extend(remaining[:needed])
        
        # 按原始顺序排序
        kept_indices.sort()
        
        return kept_indices
