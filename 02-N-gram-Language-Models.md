# Lecture 02: N-gram 语言模型

> 核心主题：数据分布与KL散度、N-gram MLE、平滑技术、Perplexity 评价指标

---

## 1. 语言模型的动机

语言模型的根本任务：**为句子赋予概率**。

应用场景：
- **语音识别**：P("recognize speech") > P("wreck a nice beach")
- **机器翻译**：选择更自然的翻译
- **拼写纠正**：P("fifteen minutes") > P("fifteen minuets")

---

## 2. 数据分布框架

### 2.1 问题设定

- 真实分布 $p_{\text{data}}$ 未知
- 我们只观测到语料（样本）：$D = \{w^{(i)}\}_{i=1}^N$，$w^{(i)} \sim p_{\text{data}}$ i.i.d.
- 目标：训练模型 $p_\theta$ 使得 $p_\theta \approx p_{\text{data}}$

### 2.2 训练目标：最小化 KL 散度

$$\theta^* = \arg\min_\theta \text{KL}(p_{\text{data}} \| p_\theta)$$

### 2.3 关键定理：KL 最小化 = 最大期望对数似然

$$\arg\min_\theta \text{KL}(p_{\text{data}}\|p_\theta) \Longleftrightarrow \arg\max_\theta \mathbb{E}_{w \sim p_{\text{data}}}[\log p_\theta(w)]$$

**证明**：
$$\text{KL}(p_{\text{data}}\|p_\theta) = \underbrace{\mathbb{E}[\log p_{\text{data}}(w)]}_{\text{常数}} - \mathbb{E}[\log p_\theta(w)]$$

第一项关于 $\theta$ 为常数，所以最小化 KL = 最大化 $\mathbb{E}[\log p_\theta(w)]$。

**实践近似**（蒙特卡洛）：

$$\max_\theta \frac{1}{N}\sum_{i=1}^N \log p_\theta(w^{(i)})$$

### 2.4 KL 散度性质

- $D_{KL}(p\|q) \neq D_{KL}(q\|p)$ （不对称）
- $D_{KL}(p\|q) \geq 0$ （非负）
- $D_{KL}(p\|q) = 0$ 当且仅当 $p = q$

---

## 3. N-gram 语言模型

### 3.1 形式化定义

**词汇表**：$V = \{v_1, v_2, \ldots, v_{|V|}\}$  
**特殊符号**：$v_0 = \text{BOS}$（句首），$v_{|V|+1} = \text{EOS}$（句尾）

**语言模型定义**：概率分布 $p(w_{1:t})$，满足：
1. $p(w_{1:t}) \geq 0$
2. $\sum_{w_{1:t}} p(w_{1:t}) = 1$

**链式法则分解**：

$$p(w_{1:t}) = \prod_{i=1}^t p(w_i \mid w_{1:i-1})$$

### 3.2 N-gram 马尔可夫假设

$$p(w_i \mid w_{1:i-1}) \approx p(w_i \mid w_{i-N+1:i-1})$$

| N | 名称 | 模型 |
|---|------|------|
| 1 | Unigram | $p(w_{1:t}) = \prod_i q(w_i)$ |
| 2 | Bigram | $p(w_{1:t}) = \prod_i q(w_i \mid w_{i-1})$ |
| 3 | Trigram | $p(w_{1:t}) = \prod_i q(w_i \mid w_{i-2}, w_{i-1})$ |

**参数量**：$\sim |V|^N$（|V|=10000, N=3 → ~$10^{12}$ 参数）

### 3.3 为什么需要 EOS

句子长度 $t$ 本身是随机变量。EOS 标记生成终止时刻。

---

## 4. 最大似然估计 (MLE)

### 4.1 Unigram MLE

$$\hat{\theta}_i = \frac{C(v_i)}{\sum_j C(v_j)} = \frac{\text{该词出现次数}}{\text{总token数}}$$

**推导**：使用 Lagrange 乘子法，约束 $\sum_i \theta_i = 1$

### 4.2 Bigram MLE

$$q(v_j \mid v_i) = \frac{C(v_i, v_j)}{\sum_k C(v_i, v_k)} = \frac{C(v_i, v_j)}{C(v_i)}$$

### 4.3 一般 N-gram MLE

$$q(x_N \mid x_{1:N-1}) = \frac{C(x_{1:N})}{C(x_{1:N-1})}$$

**本质**：相对频率估计 — N-gram 出现次数除以前缀出现次数。

---

## 5. 平滑技术

### 5.1 稀疏性问题

即使在 3800 万词的语料中，**1/3 的测试 trigram 从未出现**。MLE 对未见 N-gram 赋予零概率 → 灾难性！

### 5.2 加法平滑 (Laplace)

$$q_{\text{Add}}(x_N \mid x_{1:N-1}) = \frac{\delta + C(x_{1:N})}{\delta(|V|+1) + C(x_{1:N-1})}$$

$\delta = 1$：Laplace 平滑（加一平滑）

### 5.3 线性插值

混合各阶估计：

$$q_{\text{Int}}(x_3|x_{1:2}) = \lambda_1 \cdot q_{\text{ML}}(x_3) + \lambda_2 \cdot q_{\text{ML}}(x_3|x_2) + \lambda_3 \cdot q_{\text{ML}}(x_3|x_{1:2})$$

约束 $\lambda_1 + \lambda_2 + \lambda_3 = 1$（从验证集学习）

### 5.4 Katz 回退 (Discounting)

$$q_{\text{Katz}}(x_2|x_1) = \begin{cases} \frac{C^*(x_1,x_2)}{C(x_1)} & \text{if } C(x_1,x_2) > 0 \\ \alpha(x_1) \cdot \frac{q_{\text{ML}}(x_2)}{\sum_{x_2 \in B(x_1)} q_{\text{ML}}(x_2)} & \text{otherwise} \end{cases}$$

其中 $C^* = C - \beta$（折扣计数，通常 $\beta = 0.5$）

### 5.5 Kneser-Ney 平滑 ⭐

$$q_{\text{KN}}(x_2|x_1) = \frac{\max\{C(x_1 x_2) - \delta, 0\}}{C(x_1)} + \lambda_{x_1} \cdot q_{\text{KN}}(x_2)$$

**续接概率**（词 $x_2$ 在多少不同上下文中出现）：

$$q_{\text{KN}}(x_2) = \frac{|\{x: C(x, x_2) > 0\}|}{|\{(a,b): C(a,b) > 0\}|}$$

**Modified Kneser-Ney** 是 N-gram 的 SOTA，实现在 **KenLM** 中。

### 5.6 Stupid Back-off

非正式概率分布，简单高效：

$$S(x_N|x_{1:N-1}) = \begin{cases} \frac{C(x_{1:N})}{C(x_{1:N-1})} & \text{if } C(x_{1:N}) > 0 \\ 0.4 \cdot S(x_N|x_{2:N-1}) & \text{otherwise} \end{cases}$$

### 5.7 Good-Turing 平滑

- $N_r$ = 恰好出现 $r$ 次的类型数
- 调整计数：$r^* = (r+1) \cdot N_{r+1} / N_r$
- Good-Turing 概率：$P^*_{GT}(w) = r^* / N$

---

## 6. Perplexity (困惑度)

### 6.1 定义

设测试集 $s_1, \ldots, s_m$ 总词数 $M = \sum_i |s_i|$：

$$\ell = \frac{1}{M} \sum_{i=1}^m \log_2 p(s_i)$$

$$\text{PPL}(s_{1:m}) = 2^{-\ell}$$

### 6.2 直觉解释

"模型平均需要从多少个候选中随机选择才能选对正确词"

- PPL 越低 = 模型越好
- 词表 |V| 上的均匀模型：PPL = |V|
- |V| = 1,000,000 且 PPL = 30 → 模型有效搜索空间仅 30 个候选

### 6.3 与熵/交叉熵的关系

$$H(p,q) = H(p) + D_{KL}(p\|q) \geq H(p)$$

$$\text{PPL} = 2^{H(p,q)}$$

交叉熵是真实熵的上界，PPL 是其指数形式。

### 6.4 实验结果

**GPT-2 on WikiText-2**: PPL = **24.38**

**KenLM N-gram on WikiText-103** (Modified KN smoothing):

| 模型 | PPL |
|------|-----|
| 2-gram | 518.82 |
| 3-gram | 347.78 |
| 4-gram | 302.27 |

**对比**：GPT-2 (PPL~24) 远优于 N-gram (PPL~300+)

---

## 7. KenLM 实战流程

```bash
# 训练 N-gram 模型
lmplz -o 2 --text train_bpe.txt --arpa wikitext103_o2.arpa
lmplz -o 3 --text train_bpe.txt --arpa wikitext103_o3.arpa
lmplz -o 4 --text train_bpe.txt --arpa wikitext103_o4.arpa
```

```python
import kenlm
model = kenlm.Model("wikitext103_o3.arpa")
log10_prob = model.score("this is a sentence", bos=True, eos=True)
```

---

## 8. N-gram 文本生成

从 N-gram 生成文本：
- 2-gram："He is] = Week 15 pm guitar..." （不连贯）
- 3-gram："He is the angle bis: 23 women..." （略好）
- 4-gram：（最连贯但仍很差）

**结论**：N-gram 模型无法生成连贯的长文本，固定上下文窗口限制了表达能力。

---

## 核心要点总结

1. **KL 最小化 = MLE**：最大化对数似然等价于最小化与真实分布的 KL 散度
2. **N-gram MLE 就是计数**：相对频率 $C(x_{1:N})/C(x_{1:N-1})$
3. **平滑必不可少**：即使 3800万词语料，1/3 测试 trigram 从未出现
4. **Kneser-Ney 是 N-gram SOTA**：Modified KN (KenLM) 是实用金标准
5. **更高 N 通常更好但收益递减**：2→3→4-gram PPL: 518→348→302
6. **PPL 是语言模型的标准评价指标**：越低越好
