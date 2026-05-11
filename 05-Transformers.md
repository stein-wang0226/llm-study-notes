# Lecture 05: Transformer 架构

> 核心主题：Self-Attention、Multi-Head Attention、位置编码、Encoder-Decoder、机器翻译

---

## 1. 从 RNN 到 Attention 的动机

### 1.1 RNN 的核心问题
- BPTT 本质**序列化**（O(n) 序列操作），无法并行
- 梯度连乘导致长距离依赖难以学习
- 即使 LSTM 也只能缓解，不能根本解决

### 1.2 Attention 的直觉

对句子 "Bank of the river"，$x_1$（"Bank"）的含义取决于上下文。

**思路**：通过重新加权所有位置的表示来获得更好的上下文表示：

$$z_1 = \sum_{j=1}^4 w_{1j} \cdot x_j, \quad \text{where } w_{1j} = \text{softmax}(x_1 \cdot x_j)$$

---

## 2. Self-Attention 机制 ⭐⭐

### 2.1 三个投影

给定输入 $x_i \in \mathbb{R}^d$，学习三个投影矩阵：

- **Query**: $Q(x_i) = W_q^T \cdot x_i$, $W_q \in \mathbb{R}^{d \times d_k}$
- **Key**: $K(x_i) = W_k^T \cdot x_i$, $W_k \in \mathbb{R}^{d \times d_k}$
- **Value**: $V(x_i) = W_v^T \cdot x_i$, $W_v \in \mathbb{R}^{d \times d_v}$

### 2.2 缩放点积注意力

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**为什么除以 $\sqrt{d_k}$**：防止点积过大，使 softmax 进入梯度近零区域。

### 2.3 计算流程（以位置3为例）

1. 计算分数：$s_{3j} = Q(x_3) \cdot K(x_j)$, 对所有 $j$
2. 归一化：$w_{3j} = \text{softmax}_j(s_{3j} / \sqrt{d_k})$
3. 加权求和：$z_3 = \sum_j w_{3j} \cdot V(x_j)$

---

## 3. Multi-Head Attention ⭐

### 3.1 动机
单头注意力只有一种"关注模式"。多头允许模型同时关注不同方面。

### 3.2 公式

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \cdot W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

- 每个头有独立参数：$\theta_h = [W_{h,q}, W_{h,k}, W_{h,v}]$
- 原论文 H=8, $d_k = d_v = d_{\text{model}}/H = 64$

---

## 4. Transformer Block 完整定义

函数类 $f_\theta: \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$

### Step 1: Multi-Head Attention

$$u'_i = \sum_{h=1}^H W_{c,h}^T \sum_{j=1}^n \alpha_{i,j}^{(h)} V^{(h)}(x_j)$$

### Step 2: Add & Norm (第一个)

$$u_i = \text{LayerNorm}(x_i + u'_i; \gamma_1, \beta_1)$$

### Step 3: Feed-Forward Network (FFN)

$$z'_i = W_2^T \cdot \text{ReLU}(W_1^T \cdot u_i)$$

$W_1 \in \mathbb{R}^{d \times m}$, $W_2 \in \mathbb{R}^{m \times d}$（m = 4d = 2048）

### Step 4: Add & Norm (第二个)

$$z_i = \text{LayerNorm}(u_i + z'_i; \gamma_2, \beta_2)$$

### LayerNorm 定义

$$\text{LayerNorm}(z; \gamma, \beta) = \gamma \cdot \frac{z - \mu_z}{\sigma_z} + \beta$$

---

## 5. 完整 Transformer 架构

### 5.1 超参数（原论文 base model）

| 参数 | 值 | 含义 |
|------|-----|------|
| $d_{\text{model}}$ | 512 | 模型维度 |
| $d_k = d_v$ | 64 | 注意力头维度 |
| $H$ | 8 | 注意力头数 |
| $d_{ff}$ | 2048 | FFN 内部维度 |
| $L$ | 6 | Encoder/Decoder 层数 |

### 5.2 Encoder

$$\text{Encoder}(x) = f_{\theta_6}^E \circ f_{\theta_5}^E \circ \ldots \circ f_{\theta_1}^E(x)$$

每层：Self-Attention + FFN + 2个 Add&Norm

### 5.3 Decoder

每层**三个**子层：
1. **Masked Self-Attention**（只能看到过去的位置）
2. **Cross-Attention**（Query来自 Decoder，Key/Value来自 Encoder 输出）
3. **FFN**

### 5.4 因果掩码 (Causal Mask)

```python
def subsequent_mask(size):
    # 下三角矩阵：位置 i 只能注意位置 <= i
    mask = torch.triu(torch.ones(size, size), diagonal=1) == 0
    return mask
```

---

## 6. 位置编码 (Positional Encoding)

Self-Attention 本身不含位置信息，需要显式加入。

### 6.1 正弦位置编码

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

### 6.2 性质
- 允许模型学习相对位置
- 可外推到比训练时更长的序列
- 学习型位置编码效果类似（GPT-2 使用学习型）

---

## 7. 时间复杂度对比

| 层类型 | 每层复杂度 | 序列操作 |
|--------|-----------|---------|
| Self-Attention | $O(n^2 \cdot d)$ | **O(1)** |
| Recurrent (RNN) | $O(n \cdot d^2)$ | O(n) |

**关键**：Self-Attention 可完全并行（O(1)序列操作），但对序列长度二次。当 $n < d$ 时 Self-Attention 更优。

---

## 8. 模型大小计算

对 En-De 翻译任务 (|V|=37000, L=6, d=512, m=2048, h=8):

- **嵌入层**: |V| × d = 37,000 × 512 ≈ 18.9M 参数
- **Encoder**: 6 × (4×512² + 2×1024² + LayerNorm) ≈ 18MB
- **Decoder**: 6 × (额外一层 Cross-Attention) ≈ 24MB
- **总计**: ~61.5 MB (float32 约 246 MB 内存)

---

## 9. 训练技巧

### 9.1 学习率调度（Warmup + 逆平方根衰减）

$$lr = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}^{-1.5})$$

### 9.2 Label Smoothing

分散 (1-confidence) 的概率质量到整个词表。**提高 BLEU 但降低 PPL**。

### 9.3 嵌入缩放

$$\text{embed}(x) = \text{lookup}(x) \cdot \sqrt{d_{\text{model}}}$$

### 9.4 权重共享

输入嵌入 / 输出嵌入 / pre-softmax 线性层共享同一矩阵。

---

## 10. 核心代码实现

### Scaled Dot-Product Attention

```python
def attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    return torch.matmul(p_attn, value), p_attn
```

### Multi-Head Attention

```python
class MultiHeadedAttention(nn.Module):
    def forward(self, query, key, value, mask=None):
        # 1) 线性投影: d_model => h × d_k
        q, k, v = [lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
                   for lin, x in zip(self.linears, (query, key, value))]
        # 2) 批量注意力
        x, attn = attention(q, k, v, mask=mask)
        # 3) 拼接 + 最终投影
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.linears[-1](x)
```

### Position-wise FFN

```python
class PositionwiseFeedForward(nn.Module):
    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
```

---

## 11. 实验结果 (WMT14 翻译)

### BLEU 分数

| 模型 | EN-DE | EN-FR |
|------|-------|-------|
| Transformer (base) | 27.3 | 38.1 |
| Transformer (big) | **28.4** | **41.8** |
| 之前最优集成 | 26.36 | 41.29 |

### 消融研究关键发现

- h=1 比 h=8 低 0.9 BLEU
- 减小 $d_k$ 降低质量
- Dropout=0 降 1.2 BLEU
- 学习型 vs 正弦 PE：效果几乎相同

---

## 12. Transformer 中的三种注意力

| 类型 | Query | Key/Value | 掩码 |
|------|-------|-----------|------|
| Encoder Self-Attention | Encoder | Encoder | 无（全看） |
| Decoder Masked Self-Attention | Decoder | Decoder | 因果掩码（只看过去） |
| Encoder-Decoder Cross-Attention | Decoder | Encoder | 无 |

---

## 核心要点总结

1. **Attention 替代循环**：消除序列依赖，实现完全并行训练
2. **缩放因子 $1/\sqrt{d_k}$**：防止点积过大导致 softmax 梯度消失
3. **Multi-Head**：允许模型同时从不同表示子空间的不同位置获取信息
4. **残差连接 + LayerNorm**：使深层网络可训练
5. **位置编码**：弥补 Self-Attention 缺乏位置感知的不足
6. **Warmup 学习率**：训练稳定性的关键
7. **$O(n^2 \cdot d)$ 复杂度**：Self-Attention 对长序列有二次代价（后续 Flash Attention 等优化）
