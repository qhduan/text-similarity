# Text Similarity Package

一个基于SentenceTransformers的文本相似度计算Python包，使用Qwen3-Embedding-0.6B高质量嵌入模型。

## 功能特性

1. **字符串相似度计算**: 计算两个字符串之间的语义相似度
2. **文本向量化**: 将N个字符串转换为高质量的语义向量表示
3. **相似度矩阵**: 计算N个字符串之间的完整相似度矩阵
4. **最相似文本查找**: 在候选文本中找到与查询最相似的结果
5. **文本去重**: 基于相似度阈值从N个文本中保留指定数量的不重复文本
6. **多语言支持**: 支持中文、英文等多种语言，包括跨语言相似度计算

## 安装

```bash
# 从源码安装
pip install -e .

# 或直接安装依赖
pip install -r requirements.txt
```

**依赖要求:**
- sentence-transformers >= 2.7.0
- transformers >= 4.51.0
- torch >= 1.9.0
- numpy >= 1.19.0

## 快速开始

```python
from text_similarity import TextSimilarity

# 初始化（会自动下载Qwen3-Embedding-0.6B模型）
ts = TextSimilarity()

# 计算相似度
similarity = ts.calculate_similarity("机器学习", "深度学习")
print(f"相似度: {similarity:.4f}")  # 输出: 0.85+
```

## 使用方法

### 1. 计算两个字符串的相似度

```python
from text_similarity import TextSimilarity

# 使用默认模型（Qwen3-Embedding-0.6B）
ts = TextSimilarity()

# 中文相似度
similarity = ts.calculate_similarity("今天天气真好", "今天的天气非常不错")
print(f"相似度: {similarity:.4f}")

# 英文相似度
similarity = ts.calculate_similarity("Hello world", "Hi there world")
print(f"相似度: {similarity:.4f}")

# 跨语言相似度计算
similarity = ts.calculate_similarity("你好世界", "Hello world")
print(f"跨语言相似度: {similarity:.4f}")
```

### 2. 获取字符串向量

```python
from text_similarity import TextSimilarity

ts = TextSimilarity()

# 获取多个文本的向量表示
texts = ["机器学习", "深度学习", "人工智能", "自然语言处理"]
vectors = ts.get_vectors(texts)

print(f"向量形状: {vectors.shape}")  # 输出: (4, 768)
print(f"第一个文本的向量维度: {vectors[0].shape}")  # 输出: (768,)

# 可以使用这些向量进行各种下游任务
# 如聚类、分类、检索等
```

### 3. 获取相似度矩阵

```python
from text_similarity import TextSimilarity

ts = TextSimilarity()

texts = ["机器学习", "深度学习", "人工智能", "数据科学"]
similarity_matrix = ts.get_similarity_matrix(texts)

print(f"相似度矩阵形状: {similarity_matrix.shape}")  # 输出: (4, 4)
print(similarity_matrix)

# 矩阵示例输出（对称矩阵，对角线为1.0）:
# [[1.00, 0.85, 0.78, 0.72],
#  [0.85, 1.00, 0.80, 0.75],
#  [0.78, 0.80, 1.00, 0.70],
#  [0.72, 0.75, 0.70, 1.00]]

# 查看特定文本对的相似度
print(f"'机器学习' vs '深度学习': {similarity_matrix[0][1]:.4f}")
```

### 4. 查找最相似的文本

```python
from text_similarity import TextSimilarity

ts = TextSimilarity()

# 定义查询和候选文本
query = "机器学习算法"
candidates = [
    "深度学习神经网络",
    "云计算技术",
    "人工智能应用",
    "数据库管理",
    "自然语言处理"
]

# 找出最相似的前3个
results = ts.find_most_similar(query, candidates, top_k=3)

print(f"与 '{query}' 最相似的文本:")
for text, score, idx in results:
    print(f"  {text}: {score:.4f} (索引: {idx})")

# 输出示例:
# 与 '机器学习算法' 最相似的文本:
#   深度学习神经网络: 0.8234 (索引: 0)
#   人工智能应用: 0.7892 (索引: 2)
#   自然语言处理: 0.7654 (索引: 4)
```

### 5. 文本去重（新功能）

```python
from text_similarity import TextSimilarity

ts = TextSimilarity()

# 输入N个文本，其中有些文本内容相似
texts = [
    "机器学习是人工智能的分支",
    "深度学习是机器学习的方法",
    "机器学习属于人工智能领域",  # 与第1个相似
    "自然语言处理很重要",
    "NLP是自然语言处理的简称",  # 与第4个相似
    "计算机视觉是AI的应用",
    "深度学习使用神经网络",  # 与第2个相似
]

# 从7个文本中保留4个不重复的文本
# similarity_threshold=0.85 表示相似度超过0.85的文本被认为是重复的
kept_indices = ts.deduplicate_texts(texts, keep_count=4, similarity_threshold=0.85)

print(f"保留的文本索引: {kept_indices}")
print(f"\n保留的文本:")
for idx in kept_indices:
    print(f"  [{idx}] {texts[idx]}")

# 输出示例:
# 保留的文本索引: [0, 1, 3, 5]
# 保留的文本:
#   [0] 机器学习是人工智能的分支
#   [1] 深度学习是机器学习的方法
#   [3] 自然语言处理很重要
#   [5] 计算机视觉是AI的应用
```

## 使用不同的模型

```python
from text_similarity import TextSimilarity

# 使用默认的Qwen3-Embedding-0.6B模型
ts = TextSimilarity()

# 使用其他多语言模型
ts = TextSimilarity(model_name='paraphrase-multilingual-mpnet-base-v2')

# 使用flash attention加速（需要支持的硬件）
ts = TextSimilarity(
    model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
    tokenizer_kwargs={"padding_side": "left"}
)
```

## 推荐的模型

| 模型名称 | 维度 | 特点 |
|---------|------|------|
| `Qwen/Qwen3-Embedding-0.6B` | 768 | 默认模型，高质量中英文嵌入，性能优异 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 轻量快速，支持50+语言 |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 更高精度，支持50+语言 |

## 应用场景

- **文本检索**: 在大量文档中找到与查询最相关的内容
- **问答系统**: 匹配问题和答案
- **文本去重**: 识别并删除重复或高度相似的文本
- **语义搜索**: 基于语义而非关键词的搜索
- **推荐系统**: 基于内容的文本推荐
- **聚类分析**: 将相似文本分组
- **跨语言匹配**: 在不同语言间找到相似内容
- **内容审核**: 检测相似或重复的内容

## 性能优化建议

1. **批量处理**: 使用 `get_vectors()` 一次性处理多个文本比多次调用 `calculate_similarity()` 更高效
2. **GPU加速**: 如果有GPU，SentenceTransformers会自动使用
3. **Flash Attention**: 在支持的硬件上启用flash attention可以显著加速
4. **模型选择**: 根据需求在精度和速度之间权衡

```python
# 高效的批量相似度计算
ts = TextSimilarity()
texts = ["文本1", "文本2", "文本3", ...]  # 大量文本

# 一次性获取所有向量
vectors = ts.get_vectors(texts)

# 然后使用向量进行各种计算
# 比多次调用 calculate_similarity 快得多
```

## API参考

### TextSimilarity类

#### `__init__(model_name=None, model_kwargs=None, tokenizer_kwargs=None)`
初始化文本相似度计算器

#### `calculate_similarity(text1, text2) -> float`
计算两个字符串的相似度，返回[0, 1]范围的分数

#### `get_vectors(texts) -> np.ndarray`
返回文本列表的向量表示，形状为(N, M)

#### `get_similarity_matrix(texts) -> np.ndarray`
返回文本列表的相似度矩阵，形状为(N, N)

#### `find_most_similar(query, candidates, top_k=5) -> List[tuple]`
查找最相似的文本，返回(文本, 分数, 索引)的列表

#### `deduplicate_texts(texts, keep_count, similarity_threshold=0.9) -> List[int]`
文本去重，返回保留文本的索引列表

## 注意事项

- 首次运行时会自动下载Qwen3-Embedding-0.6B模型（约1.2GB）
- 模型会缓存在本地，后续使用无需重新下载
- 建议在稳定的网络环境下首次运行
- 使用GPU可以显著提升处理速度
