# Lecture 01: Tokenization (文本预处理)

> 核心主题：正则表达式、数据探索 (Heap定律)、最小编辑距离、现代分词方法 (BPE/WordPiece/Unigram)

---

## 1. LLM 作为概率模型

### 1.1 核心数学表述

LLM 建模条件概率：

$$p_\theta(w_{n+1} \mid w_1, w_2, \ldots, w_n) = \frac{p_\theta(w_1, w_2, \ldots, w_n, w_{n+1})}{p_\theta(w_1, w_2, \ldots, w_n)}$$

- **Prompt** = $[w_1, w_2, \ldots, w_n]$
- **Response** = $[w_{n+1}, \ldots]$

**关键洞察**：如果能学到对任意长度序列的概率分配 $p_\theta(w_1, \ldots, w_k)$，条件概率自然得到。

### 1.2 生成控制参数

| 参数 | 作用 |
|------|------|
| `temperature` | T<1 更确定，T>1 更随机 |
| `top_p` | 核采样，累积概率阈值 |
| `top_k` | 只从 top-k 概率的 token 中采样 |
| 确定性输出 | `temperature=0.0, top_p=1.0, top_k=0` |

---

## 2. Unicode 与文本编码

- **ASCII**: 128 字符，单字节
- **UTF-8**: 变长编码 (1-4 字节)，中文字符用 3 字节
- `string.encode('utf-8')` → 字节序列
- `bytes.decode('utf-8')` → 字符串

---

## 3. 正则表达式 (RE)

| 模式 | 含义 | 示例 |
|------|------|------|
| `[wW]` | 析取 | 匹配 `woodchuck` 或 `Woodchuck` |
| `[0-9]` | 数字范围 | 匹配任意单个数字 |
| `[^A-Z]` | 否定 | 匹配非大写字母 |
| `\|` | 或 | `groundhog\|woodchuck` |
| `?` | 可选 (0或1) | `colou?r` 匹配 `color`/`colour` |
| `*` | 0次或多次 | |
| `+` | 1次或多次 | |
| `.` | 任意字符 | |
| `\b` | 词边界 | `\b[tT]he\b` 精确匹配 "the" |

**实用正则**：`[A-Za-z]+(?:'[A-Za-z]+)?` — 匹配英文单词含缩写（如 "don't"）

---

## 4. 分词方法分类

### 4.1 自顶向下 (Top-down, 传统方法)
- 规则定义分割方式
- 代表：spaCy (处理标点、中文分词、NER)
- 局限：无法处理无空格语言，规则复杂

### 4.2 自底向上 (Bottom-up, 现代方法)
- 基于字符统计构建 subword token
- 代表：BPE、WordPiece、Unigram
- 所有现代 LLM 使用此类方法

---

## 5. Heap 定律 (Herdan's Law)

### 公式

$$|V| = k \cdot T^\beta$$

| 参数 | 含义 | 典型范围 |
|------|------|----------|
| $\|V\|$ | 词汇表大小 (types) | |
| $T$ | 总 token 数 (instances) | |
| $k$ | 常数 | [10, 100] |
| $\beta$ | 增长指数 | [0.4, 0.6], 大语料 [0.67, 0.75] |

### 拟合方法

取对数线性化：$\log|V| = \log k + \beta \log T$

在 log-log 坐标下线性回归得到 $k$ 和 $\beta$。

### 含义

词汇量随语料增长呈亚线性增长（新词不断出现，但增速减缓）。

---

## 6. 现代 Tokenization 方法

### 6.1 BPE (Byte Pair Encoding)

**算法步骤**：
1. 以所有单个字符作为初始 token
2. 找出最频繁的相邻 token 对
3. 将该对合并为新 token
4. 重复直到达到目标词汇表大小

**使用者**：GPT-2, GPT-3, GPT-4

**tiktoken 编码器**：
- `cl100k_base`: GPT-3.5-turbo, GPT-4
- `o200k_base`: GPT-4o 系列

**训练代码**：
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(vocab_size=30000, 
                     special_tokens=["[UNK]","[PAD]","[BOS]","[EOS]","[MASK]"])
tokenizer.train_from_iterator(data_iterator, trainer=trainer)
```

### 6.2 WordPiece

**流程**：
1. Normalization: 去空格、去重音、小写
2. Pre-tokenization: 按空格分割
3. Model: WordPiece（用 `##` 前缀标记续接子词）
4. Post-processing: 添加 `[CLS]`, `[SEP]`

**使用者**：BERT, GNMT

**示例**：`"Tokenizers"` → `["tok", "##eni", "##zer", "##s"]`

### 6.3 Unigram

**使用者**：ALBERT, T5

基于统计的子词选择，从大词表逐步删减。

### 对比表

| 方法 | 论文年份 | 采用模型 | 特点 |
|------|---------|---------|------|
| BPE | 1994 | GPT系列 | 贪心合并最频繁对 |
| WordPiece | 1994 | BERT | 类似BPE，`##`前缀 |
| Unigram | 2018 | T5/ALBERT | 概率性，从大到小删减 |

### 6.4 局限性

通用 tokenizer 对领域文本（化学SMILES、基因序列）可能产生语义破坏性的切分。例如阿司匹林的 SMILES `CC(=O)OC1=CC=CC=C1C(O)=O` 会被切成多个无意义 token。

---

## 7. 最小编辑距离 (Minimum Edit Distance, MED)

### 7.1 定义

给定字符串 $X = [x_1, \ldots, x_n]$ 和 $Y = [y_1, \ldots, y_m]$，MED 是将 X 转换为 Y 的最小编辑操作代价。

| 操作 | 代价 |
|------|------|
| 插入 (Ins) | 1 |
| 删除 (Del) | 1 |
| 替换 (Sub) | 2 |

### 7.2 动态规划

**子问题**：$D[i,j]$ = $X_i$ 到 $Y_j$ 的 MED

**基础情况**：
- $D[i, 0] = i$ （删除所有字符）
- $D[0, j] = j$ （插入所有字符）

**递推关系**：

$$D[i,j] = \min \begin{cases} D[i-1, j] + 1 & \text{(删除 } x_i\text{)} \\ D[i, j-1] + 1 & \text{(插入 } y_j\text{)} \\ D[i-1, j-1] + \text{sub}(x_i, y_j) & \text{(替换)} \end{cases}$$

其中 $\text{sub}(x_i, y_j) = 0$ 若 $x_i = y_j$，否则 $= 2$。

**时间复杂度**：$O(n \cdot m)$

### 7.3 经典示例

INTENTION → EXECUTION: MED = 8

```
I N T E N T I O N
↓                    Del I (1)
  N T E N T I O N
    ↓                Sub N→E (2)
  E T E N T I O N
      ↓              Sub T→X (2)
  E X E N T I O N
        ↓            Ins C (1)
  E X E C N T I O N
          ↓          Sub N→U (2)
  E X E C U T I O N
Total: 1+2+2+1+2 = 8
```

### 7.4 实现代码

```python
def minimum_edit_distance(source, target):
    n, m = len(source), len(target)
    d_mat = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1): d_mat[i, 0] = i
    for j in range(1, m + 1): d_mat[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if source[i-1] == target[j-1] else 2
            d_mat[i][j] = min(
                d_mat[i-1][j] + 1,       # 删除
                d_mat[i][j-1] + 1,       # 插入
                d_mat[i-1][j-1] + sub    # 替换/匹配
            )
    return d_mat[n, m]
```

### 7.5 DP 在经典 NLP 中的应用

1. 最小编辑距离 (MED)
2. HMM (隐马尔可夫模型)
3. CKY 句法分析
4. CRF (Viterbi) 用于 NER

---

## 核心要点总结

1. **LLM 是概率序列模型**，通过条件概率生成文本
2. **分词从规则到统计**演进：BPE 是当前主流
3. **Heap 定律**描述词汇量与语料量的经验关系（亚线性增长）
4. **MED + DP** 是 NLP 的基础算法范式，同一思想支撑 HMM/CKY/CRF
5. **UTF-8** 是现代多语言文本标准，理解字节级表示是理解 byte-level BPE 的前提
