# Lecture 04: 神经语言模型

> 核心主题：NPLM (Bengio 2003)、RNN、BPTT、梯度消失/爆炸、LSTM

---

## 1. 神经概率语言模型 (NPLM, Bengio et al. 2003)

### 1.1 核心公式

$$\hat{P}(w_t \mid w_{t-1}, \ldots, w_{t-N+1}) = \frac{e^{y_{w_t}}}{\sum_i e^{y_i}}$$

**未归一化 log 概率**：
$$y = b + Wx + U \cdot \tanh(d + Hx)$$

**输入表示**（拼接嵌入）：
$$x = (C(w_{t-1}), C(w_{t-2}), \ldots, C(w_{t-N+1})) \in \mathbb{R}^{(N-1) \times m}$$

### 1.2 参数

| 参数 | 维度 | 角色 |
|------|------|------|
| $C$ | $\mathbb{R}^{\|V\| \times m}$ | 词嵌入矩阵 |
| $H$ | $\mathbb{R}^{h \times (N-1)m}$ | 输入→隐藏层权重 |
| $d$ | $\mathbb{R}^h$ | 隐藏层偏置 |
| $U$ | $\mathbb{R}^{\|V\| \times h}$ | 隐藏→输出权重 |
| $W$ | $\mathbb{R}^{\|V\| \times (N-1)m}$ | 直接连接权重（可选） |
| $b$ | $\mathbb{R}^{\|V\|}$ | 输出偏置 |

### 1.3 参数量对比

| 模型 | 参数量级 |
|------|---------|
| N-gram | $O(\|V\|^N)$ — 指数级 |
| NPLM | $O(\|V\| \cdot N \cdot m)$ — 线性级 |

**关键优势**：通过共享嵌入，NPLM 参数量从指数降为线性！

### 1.4 架构图

```
[输出层] ← softmax(y) → P(w_t = i | context)
     ↑
(Wx + b) + U·tanh(Hx + d)    ← "主要计算在此"
     ↑              ↑
  [直接连接]    [tanh 隐藏层]
     ↑              ↑
     x = [C(w_{t-n+1}), ..., C(w_{t-1})]
     ↑
[查表 C] ← 共享参数
     ↑
index for w_{t-n+1}, ..., w_{t-1}
```

### 1.5 实验结果

- 在 Shakespeare 上训练：正确学到 "the cat sat" → "on" (概率 60.25%)
- WikiText-2 训练（GPT-2 tokenizer, vocab=50257, emb=64, hidden=128）

---

## 2. 循环神经网络 (RNN)

### 2.1 RNN 方程

$$h_t = \phi_h(W_{xh}^T x_t + W_{hh}^T h_{t-1} + b_h)$$
$$o_t = W_{yh} \cdot h_t + b_y$$
$$\hat{y}_t = \text{softmax}(o_t)$$

- $\phi_h$：tanh 或 ReLU
- $W_{hh}$：循环权重（记忆单元）
- 时间展开：$h_0 \to h_1 \to h_2 \to \ldots \to h_T$

### 2.2 损失函数

$$L(\hat{y}, y) = -\sum_{t=1}^T y_t \log \hat{y}_t = -\sum_{t=1}^T L_t$$

---

## 3. 时间反向传播 (BPTT)

### 3.1 对 $W_{yh}$ 的梯度（简单）

$$\frac{\partial L}{\partial W_{yh}} = \sum_{t=1}^T \frac{\partial L_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial o_t} \cdot \frac{\partial o_t}{\partial W_{yh}}$$

### 3.2 对 $W_{hh}$ 的梯度（关键难点）⭐

$W_{hh}$ 影响所有未来隐藏状态，需要累加所有路径：

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \sum_{k=1}^t \frac{\partial L_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial h_k} \cdot \frac{\partial h_k}{\partial W_{hh}}$$

**核心项**：

$$\frac{\partial h_t}{\partial h_k} = \prod_{j=k}^{t-1} \frac{\partial h_{j+1}}{\partial h_j}$$

### 3.3 Jacobian 矩阵

$$\frac{\partial h_t}{\partial h_{t-1}} = \text{Diag}(\phi'_h) \cdot W_{hh}$$

---

## 4. 梯度消失与爆炸 ⭐⭐

### 4.1 范数分析

$$\left\|\frac{\partial h_t}{\partial h_k}\right\| \leq \prod_{i=k}^{t-1} \|\text{Diag}(\phi'_h)\| \cdot \|W_{hh}\| \leq (\gamma_h \cdot \gamma_w)^{t-k}$$

其中：
- $\gamma_h = \|\text{Diag}(\phi'_h)\| \leq 1$（tanh 导数最大为1，sigmoid 最大为0.25）
- $\gamma_w = \|W_{hh}\|$

### 4.2 两种情况

| 条件 | 结果 | 后果 |
|------|------|------|
| $\gamma_h \cdot \gamma_w < 1$ | $(\gamma_h \gamma_w)^{t-k} \to 0$ | **梯度消失** — 无法学习长距离依赖 |
| $\gamma_h \cdot \gamma_w > 1$ | $(\gamma_h \gamma_w)^{t-k} \to \infty$ | **梯度爆炸** — 训练不稳定 |

### 4.3 解决方案

1. **梯度裁剪**（解决爆炸）：若 $\|g\| > \text{threshold}$，则 $g \leftarrow \frac{\text{threshold}}{\|g\|} \cdot g$
2. **Leaky Integration Units**
3. **LSTM / GRU**（解决消失）

---

## 5. LSTM (长短期记忆网络) ⭐⭐

### 5.1 四个组件

**① 遗忘门**（决定丢弃什么信息）：
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**② 输入门 + 候选值**（决定存储什么新信息）：
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**③ 细胞状态更新**：
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**④ 输出门**：
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

### 5.2 为什么 LSTM 解决梯度消失

细胞状态 $C_t$ 使用**加法**更新（非乘法），梯度可以沿着"恒定误差传送带"无阻碍流动：

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

当 $f_t \approx 1$（遗忘门开启），梯度 $\frac{\partial C_t}{\partial C_{t-1}} \approx 1$，不衰减。

### 5.3 实验结果 (WikiText-2)

| Epoch | Train PPL | Valid PPL |
|-------|-----------|-----------|
| 1 | 1521.01 | 971.39 |
| 5 | 398.48 | 433.80 |
| 10 | 212.12 | 317.88 |

**测试集 PPL = 326.15**（2层 LSTM, hidden=200, 10个epoch）

---

## 6. PyTorch 自动微分 (Autograd)

### 6.1 核心概念

```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
out.backward()      # 反向传播
print(x.grad)       # tensor([[4.5, 4.5], [4.5, 4.5]])
```

**验证**：$\text{out} = \frac{3}{4}\sum(x+2)^2$，$\frac{\partial \text{out}}{\partial x} = \frac{3}{2}(x+2) = \frac{3}{2} \times 3 = 4.5$ ✓

### 6.2 Tensor/NumPy 共享内存

```python
a = torch.ones(5)
b = a.numpy()       # 共享内存！
a.add_(1)           # a 和 b 都变成 [2,2,2,2,2]
```

---

## 7. Micrograd（手写自动微分引擎）

### Value 类核心思想

```python
class Value:
    def __add__(self, other):
        out = Value(self.data + other.data)
        def _backward():
            self.grad += out.grad    # ∂L/∂self = ∂L/∂out × 1
            other.grad += out.grad   # ∂L/∂other = ∂L/∂out × 1
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data)
        def _backward():
            self.grad += other.data * out.grad  # ∂L/∂self = other × ∂L/∂out
            other.grad += self.data * out.grad  # ∂L/∂other = self × ∂L/∂out
        out._backward = _backward
        return out
```

**backward()**: 拓扑排序 → 逆序应用链式法则

---

## 8. NPLM 完整实现

```python
class BengioNPLM(nn.Module):
    def __init__(self, config):
        self.emb = nn.Embedding(vocab_size, embedding_dim)       # C 矩阵
        self.hidden = nn.Linear(context_size * emb_dim, hidden_dim)  # H
        self.hidden_to_vocab = nn.Linear(hidden_dim, vocab_size)  # U
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))  # b
        self.direct = nn.Linear(context_size * emb_dim, vocab_size)  # W (可选)
    
    def forward(self, input_ids):
        x = self.emb(input_ids).reshape(B, -1)          # 拼接嵌入
        h = torch.tanh(self.hidden(x))                   # tanh(Hx + d)
        logits = self.hidden_to_vocab(h) + self.output_bias  # U·h + b
        logits = logits + self.direct(x)                 # + Wx
        return logits
```

---

## 核心要点总结

1. **NPLM 比 N-gram 参数量从指数级降为线性级**，通过共享嵌入实现泛化
2. **RNN 的 BPTT 本质上是序列化的**，无法并行化 $\prod_{j=k}^t \partial h_{j+1}/\partial h_j$
3. **梯度消失/爆炸**是 vanilla RNN 的核心缺陷，源于 Jacobian 矩阵的连乘
4. **LSTM 通过加法细胞状态更新**解决梯度消失：$C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t$
5. **梯度裁剪**是稳定 RNN/LSTM 训练的必备技巧（代码中 CLIP=0.25）
6. **BPTT 的序列化本质**是 Transformer 试图解决的核心问题（下一讲）
