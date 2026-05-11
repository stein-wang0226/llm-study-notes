# Lecture 03: 词嵌入 (Embeddings)

> 核心主题：文本分类 (NB/LR)、TF-IDF、PPMI、Word2Vec (Skip-gram + 负采样)、现代嵌入模型

---

## 1. 文本分类方法

### 1.1 朴素贝叶斯 (Naive Bayes, 生成模型)

$$c_{\text{MAP}} = \arg\max_{c \in C} P(d|c) \cdot P(c)$$

- **假设**：词之间条件独立
- **参数**：$\theta = [\hat{P}(c), \hat{P}(v|c)]$
- **实验结果** (20类新闻组)：
  - 训练精度：97.4%
  - 测试精度：**80.4%**

### 1.2 逻辑回归 (Logistic Regression, 判别模型)

$$P_\theta(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

**Sigmoid 关键性质**：$1 - \sigma(z) = \sigma(-z)$

**Softmax (多分类)**：
$$\text{softmax}(z_c) = \frac{e^{z_c}}{\sum_{j=1}^K e^{z_j}}$$

**二元交叉熵损失**：
$$\mathcal{L} = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$

**实验结果** (IMDB 情感分析, TF-IDF + LR)：测试精度 **90.2%**

---

## 2. 词表示方法演进

| 方法 | 类型 | 核心思想 |
|------|------|----------|
| Bag of Words | 计数 | 原始词频向量 |
| TF-IDF | 计数 | 频率 × 文档稀有度加权 |
| PPMI | 计数 | 观测/期望共现比的对数 |
| Word2Vec | 学习 | 从词预测上下文（或反之） |
| Transformer Embeddings | 学习 | 深度注意力网络的上下文嵌入 |

---

## 3. TF-IDF

$$\text{TF}_{t,d} = \log_{10}(\text{count}(t, d) + 1)$$
$$\text{IDF}_t = \log_{10}\frac{N}{\text{DF}_t}$$
$$w_{t,d} = \text{TF}_{t,d} \cdot \text{IDF}_t$$

- TF：词频（对数缩放）
- IDF：逆文档频率（越稀有越重要）

---

## 4. PPMI (正逐点互信息)

$$\text{PMI}(w_1, w_2) = \log\frac{p(w_1, w_2)}{p(w_1) \cdot p(w_2)}$$

$$\text{PPMI}(w_1, w_2) = \max(\text{PMI}(w_1, w_2), 0)$$

其中：
- $p_{i,j} = f_{i,j} / \sum_{i,j} f_{i,j}$ （联合概率）
- $p_{i*} = \sum_j f_{i,j} / \sum_{i,j} f_{i,j}$ （边缘概率）

---

## 5. Word2Vec — Skip-gram with Negative Sampling (SGNS) ⭐

### 5.1 核心思想

两个嵌入矩阵：
- $U = [u_1, \ldots, u_{|V|}]$：目标词/中心词嵌入
- $V = [v_1, \ldots, v_{|V|}]$：上下文词嵌入

### 5.2 概率定义

**正样本**（c 是 w 的真实上下文）：
$$P(D=1 \mid w, c) = \sigma(u_w^T \cdot v_c)$$

**负样本**（c 不是 w 的上下文）：
$$P(D=0 \mid w, c) = 1 - \sigma(u_w^T \cdot v_c) = \sigma(-u_w^T \cdot v_c)$$

### 5.3 目标函数

对每个正样本对 $(w, c_{\text{pos}})$，采样 $k$ 个负上下文 $c_{\text{neg}_1}, \ldots, c_{\text{neg}_k}$：

$$\mathcal{L} = -\left[\log \sigma(u_w^T \cdot v_{c_{\text{pos}}}) + \sum_{i=1}^k \log \sigma(-u_w^T \cdot v_{c_{\text{neg}_i}})\right]$$

### 5.4 全局目标

$$J(\theta) = -\frac{1}{T} \sum_{t=1}^T \sum_{-k \leq j \leq k, j \neq 0} \log p(c_{t+j} \mid w_t; \theta)$$

### 5.5 关键超参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| Window size | 上下文窗口 | 5-20 |
| Negative samples (k) | 负样本数 | 大数据集 2-5，小数据集 5-20 |
| 嵌入维度 | 向量大小 | 100-300 |
| α (噪声分布指数) | 平滑负采样分布 | **0.75** |

### 5.6 噪声分布

$$P_N(w) = \left(\frac{U(w)}{Z}\right)^{0.75}$$

- 提升稀有词的采样概率
- 降低高频词的采样概率

### 5.7 训练代码

```python
import gensim
model = gensim.models.Word2Vec(
    sentences=sentences,
    vector_size=100,      # 嵌入维度
    window=5,             # 上下文窗口
    min_count=5,          # 最低词频
    sg=1,                 # 1=Skip-gram, 0=CBOW
    negative=5,           # 负样本数
    ns_exponent=0.75,     # α=0.75
    epochs=5
)
# 获取词向量
king_vec = model.wv["king"]  # 100维向量
```

### 5.8 网络结构

```
输入层 (one-hot, N维)
    ↓ W_in (嵌入矩阵)
隐藏层 (h, M维)
    ↓ W_out (上下文矩阵)
输出层 (softmax/负采样, V维)
```

**重要**：完整 softmax 需要对整个词汇表求和（太贵），负采样通过只计算 k+1 个 token 来近似。

---

## 6. 现代嵌入模型

### 6.1 Qwen3 Embedding (0.6B)

```python
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
# 使用 last_token_pool 提取句子嵌入
# L2 归一化后用余弦相似度
```

**实验结果**（语义相似度矩阵）：

|  | Doc1 (北京) | Doc2 (引力) |
|--|-----------|-----------|
| Query1 (中国首都) | **0.765** | 0.155 |
| Query2 (解释引力) | 0.135 | **0.693** |

正确匹配得分远高于不相关对。

### 6.2 与 Word2Vec 的关键区别

| 特性 | Word2Vec | Transformer Embedding |
|------|----------|----------------------|
| 上下文 | 静态（一词一向量） | 动态（相同词不同上下文有不同表示） |
| 训练目标 | 预测上下文 | MLM 或对比学习 |
| 维度 | 100-300 | 768-4096 |
| 支持指令 | 否 | 是（如 Qwen3-Embedding） |

---

## 7. SGNS 与 PMI 矩阵的理论联系

**Levy & Goldberg (2014)** 证明：SGNS 目标隐式分解了 PMI 矩阵。

即：学习型方法（Word2Vec）和计数型方法（PPMI）在理论上是等价的！

---

## 核心要点总结

1. **词表示演进**：稀疏计数 (BoW/TF-IDF/PPMI) → 密集学习 (Word2Vec) → 上下文化 (Transformer)
2. **Word2Vec 的负采样**使训练从 O(|V|) 降到 O(k)，k 通常为 5
3. **α=0.75** 噪声分布指数是关键实现细节
4. **KL 最小化 = 最大似然**是贯穿所有模型的统一框架
5. **现代嵌入模型**可通过指令控制嵌入语义方向
6. **NB (~80%) < LR+TF-IDF (~90%)** 在文本分类上的经验对比
