"""
基于阿里云百炼API的文本相似度计算模块
"""

import os
import asyncio
import aiohttp
import numpy as np
from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity


class TextSimilarityBailian:
    """
    文本相似度计算类 - 使用阿里云百炼Embedding API
    
    需要设置环境变量 BAILIAN_TOKEN
    """
    
    def __init__(
        self, 
        api_token: Optional[str] = None,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
        encoding_format: str = "float"
    ):
        """
        初始化TextSimilarityBailian对象
        
        Args:
            api_token (str, optional): 百炼API Token，如果不提供则从环境变量BAILIAN_TOKEN读取
            model (str): 模型名称，默认为text-embedding-v4
            dimensions (int): 向量维度，默认为1024
            encoding_format (str): 编码格式，默认为float
        
        Raises:
            ValueError: 如果未提供api_token且环境变量BAILIAN_TOKEN未设置
        
        Examples:
            >>> ts = TextSimilarityBailian()
            >>> ts = TextSimilarityBailian(api_token="your-token", dimensions=512)
        """
        self.api_token = api_token or os.environ.get("BAILIAN_TOKEN")
        if not self.api_token:
            raise ValueError(
                "未找到API Token。请通过参数提供api_token或设置环境变量BAILIAN_TOKEN"
            )
        
        self.model = model
        self.dimensions = dimensions
        self.encoding_format = encoding_format
        self.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
    
    async def _get_embedding(self, text: str) -> List[float]:
        """
        异步获取单个文本的向量嵌入
        
        Args:
            text (str): 输入文本
        
        Returns:
            List[float]: 文本的向量嵌入
        """
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "input": text,
            "dimensions": str(self.dimensions),
            "encoding_format": self.encoding_format
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.api_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["data"][0]["embedding"]
                    else:
                        error_text = await response.text()
                        raise Exception(
                            f"API请求失败，状态码: {response.status}, 错误信息: {error_text}"
                        )
            except Exception as e:
                raise Exception(f"请求发生错误: {str(e)}")
    
    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        异步批量获取多个文本的向量嵌入
        
        Args:
            texts (List[str]): 文本列表
        
        Returns:
            List[List[float]]: 向量嵌入列表
        """
        tasks = [self._get_embedding(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个字符串之间的相似度（同步接口）
        
        Args:
            text1 (str): 第一个字符串
            text2 (str): 第二个字符串
        
        Returns:
            float: 相似度分数，范围为[0, 1]，1表示完全相同
        
        Examples:
            >>> ts = TextSimilarityBailian()
            >>> similarity = ts.calculate_similarity("你好世界", "你好中国")
            >>> print(f"相似度: {similarity:.4f}")
        """
        return asyncio.run(self.calculate_similarity_async(text1, text2))
    
    async def calculate_similarity_async(self, text1: str, text2: str) -> float:
        """
        计算两个字符串之间的相似度（异步接口）
        
        Args:
            text1 (str): 第一个字符串
            text2 (str): 第二个字符串
        
        Returns:
            float: 相似度分数，范围为[0, 1]
        """
        embeddings = await self._get_embeddings_batch([text1, text2])
        
        vec1 = np.array(embeddings[0]).reshape(1, -1)
        vec2 = np.array(embeddings[1]).reshape(1, -1)
        
        similarity = cosine_similarity(vec1, vec2)[0][0]
        return float(similarity)
    
    def get_vectors(self, texts: List[str]) -> np.ndarray:
        """
        返回N个字符串的向量表示（同步接口）
        
        Args:
            texts (List[str]): 字符串列表
        
        Returns:
            np.ndarray: 形状为(N, M)的向量矩阵
                - N为字符串数量
                - M为特征维度（默认1024）
        
        Examples:
            >>> ts = TextSimilarityBailian()
            >>> texts = ["机器学习", "深度学习", "人工智能"]
            >>> vectors = ts.get_vectors(texts)
            >>> print(f"向量形状: {vectors.shape}")
        """
        return asyncio.run(self.get_vectors_async(texts))
    
    async def get_vectors_async(self, texts: List[str]) -> np.ndarray:
        """
        返回N个字符串的向量表示（异步接口）
        
        Args:
            texts (List[str]): 字符串列表
        
        Returns:
            np.ndarray: 形状为(N, M)的向量矩阵
        """
        if not texts:
            raise ValueError("文本列表不能为空")
        
        embeddings = await self._get_embeddings_batch(texts)
        return np.array(embeddings)
    
    def get_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        返回N个字符串的相似度矩阵（同步接口）
        
        Args:
            texts (List[str]): 字符串列表
        
        Returns:
            np.ndarray: 形状为(N, N)的相似度矩阵
        
        Examples:
            >>> ts = TextSimilarityBailian()
            >>> texts = ["机器学习", "深度学习", "人工智能", "数据科学"]
            >>> matrix = ts.get_similarity_matrix(texts)
            >>> print(f"相似度矩阵形状: {matrix.shape}")
        """
        return asyncio.run(self.get_similarity_matrix_async(texts))
    
    async def get_similarity_matrix_async(self, texts: List[str]) -> np.ndarray:
        """
        返回N个字符串的相似度矩阵（异步接口）
        
        Args:
            texts (List[str]): 字符串列表
        
        Returns:
            np.ndarray: 形状为(N, N)的相似度矩阵
        """
        if not texts:
            raise ValueError("文本列表不能为空")
        
        vectors = await self.get_vectors_async(texts)
        similarity_matrix = cosine_similarity(vectors, vectors)
        
        return similarity_matrix
    
    def find_most_similar(self, query: str, candidates: List[str], top_k: int = 5) -> List[tuple]:
        """
        在候选文本中找到与查询文本最相似的top_k个结果（同步接口）
        
        Args:
            query (str): 查询文本
            candidates (List[str]): 候选文本列表
            top_k (int): 返回最相似的前k个结果，默认为5
        
        Returns:
            List[tuple]: 包含(文本, 相似度分数, 索引)的列表，按相似度降序排列
        
        Examples:
            >>> ts = TextSimilarityBailian()
            >>> query = "机器学习算法"
            >>> candidates = ["深度学习", "神经网络", "数据挖掘"]
            >>> results = ts.find_most_similar(query, candidates, top_k=2)
            >>> for text, score, idx in results:
            ...     print(f"{text}: {score:.4f}")
        """
        return asyncio.run(self.find_most_similar_async(query, candidates, top_k))
    
    async def find_most_similar_async(self, query: str, candidates: List[str], top_k: int = 5) -> List[tuple]:
        """
        在候选文本中找到与查询文本最相似的top_k个结果（异步接口）
        
        Args:
            query (str): 查询文本
            candidates (List[str]): 候选文本列表
            top_k (int): 返回最相似的前k个结果
        
        Returns:
            List[tuple]: 包含(文本, 相似度分数, 索引)的列表
        """
        if not candidates:
            raise ValueError("候选文本列表不能为空")
        
        # 获取查询和候选文本的向量
        all_texts = [query] + candidates
        vectors = await self.get_vectors_async(all_texts)
        
        query_vector = vectors[0:1]
        candidate_vectors = vectors[1:]
        
        # 计算相似度
        similarities = cosine_similarity(query_vector, candidate_vectors)[0]
        
        # 获取top_k结果
        top_k = min(top_k, len(candidates))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((candidates[idx], float(similarities[idx]), int(idx)))
        
        return results
    
    def deduplicate_texts(self, texts: List[str], keep_count: int, similarity_threshold: float = 0.9) -> List[int]:
        """
        文本去重：从N个文本中保留指定数量的文本，返回保留文本的索引（同步接口）
        
        Args:
            texts (List[str]): 输入的N个文本列表
            keep_count (int): 要保留的文本数量
            similarity_threshold (float): 相似度阈值，默认0.9
        
        Returns:
            List[int]: 保留的文本在原列表中的索引列表
        
        Examples:
            >>> ts = TextSimilarityBailian()
            >>> texts = ["机器学习很重要", "深度学习", "机器学习非常重要"]
            >>> indices = ts.deduplicate_texts(texts, keep_count=2, similarity_threshold=0.85)
            >>> print(f"保留的索引: {indices}")
        """
        return asyncio.run(self.deduplicate_texts_async(texts, keep_count, similarity_threshold))
    
    async def deduplicate_texts_async(self, texts: List[str], keep_count: int, similarity_threshold: float = 0.9) -> List[int]:
        """
        文本去重：从N个文本中保留指定数量的文本，返回保留文本的索引（异步接口）
        
        Args:
            texts (List[str]): 输入的N个文本列表
            keep_count (int): 要保留的文本数量
            similarity_threshold (float): 相似度阈值
        
        Returns:
            List[int]: 保留的文本在原列表中的索引列表
        """
        if not texts:
            raise ValueError("文本列表不能为空")
        
        if keep_count <= 0:
            raise ValueError("keep_count必须大于0")
        
        if keep_count >= len(texts):
            return list(range(len(texts)))
        
        # 获取所有文本的向量
        vectors = await self.get_vectors_async(texts)
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(vectors, vectors)
        
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
