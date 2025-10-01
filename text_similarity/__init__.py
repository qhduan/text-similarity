"""
Text Similarity Package
一个用于计算文本相似度的Python包
"""

from .similarity import TextSimilarity
from .bailian_similarity import TextSimilarityBailian

__version__ = "0.1.0"
__all__ = ["TextSimilarity", "TextSimilarityBailian"]
