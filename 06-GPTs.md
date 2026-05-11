# Lecture 06: GPT 系列 — 从零训练 GPT-2

> 核心主题：GPT 架构实现、从零预训练、Flash Attention、分布式训练 (DDP)、HellaSwag 评测

---

## 1. GPT 模型家族

| 模型 | 时间 | 论文 |
|------|------|------|
| GPT-1 | 2018.06 | "Improving Language Understanding by Generative Pre-Training" |
| GPT-2 | 2019.02 | "Language Models are Unsupervised Multitask Learners" |
| GPT-3 | 2020.05 | "Language Models are Few-Shot Learners" |
| GPT-4 | 2023.03 | "GPT-4 Technical Report" |

### GPT-2 规格

| 模型 | 层数 | 头数 | 嵌入维度 | 参数量 |
|------|------|------|---------|--------|
| gpt2 | 12 | 12 | 768 | 124M |
| gpt2-medium | 24 | 16 | 1024 | 350M |
| gpt2-large | 36 | 20 | 1280 | 774M |
| gpt2-xl | 48 | 25 | 1600 | 1558M |

---

## 2. 文本生成策略

| 策略 | 特点 | 问题 |
|------|------|------|
| Greedy Search | 每步选最高概率 | 重复循环 |
| Beam Search | 保持多个假设 | 仍然重复 |
| Beam + N-gram penalty | 禁止 N-gram 重复 | 可能语法不通 |
| Temperature Sampling | T<1更确定, T>1更随机 | T太高不连贯 |
| Top-k Sampling | 只从 top-k 中采样 | k固定，不自适应 |
| Top-p (Nucleus) | 累积概率超过 p 的最小集合 | **自适应词表大小** |

**最佳实践配置**：
```python
top_k=50, top_p=0.95, temperature=0.7, repetition_penalty=1.2
```

---

## 3. GPT-2 架构实现 ⭐⭐

### 3.1 与原始 Transformer 的区别

| 特性 | 原始 Transformer | GPT-2 |
|------|-----------------|-------|
| 架构 | Encoder-Decoder | **Decoder-only** |
| LayerNorm | Post-Norm | **Pre-Norm** |
| 位置编码 | 正弦 (固定) | **学习型** |
| 激活函数 | ReLU | **GELU** |

### 3.2 核心组件

#### GPTConfig
```python
@dataclass
class GPTConfig:
    block_size: int = 1024    # 最大序列长度
    vocab_size: int = 50257   # 50000 BPE + 256 字节 + 1 <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
```

#### CausalSelfAttention
```python
class CausalSelfAttention(nn.Module):
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)  # 一次投影得到 Q,K,V
        q, k, v = qkv.split(self.n_embd, dim=2)
        # Reshape: (B, T, n_head, head_size) → (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))
```

#### Block (Pre-Norm)
```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # 先Norm再Attention
        x = x + self.mlp(self.ln_2(x))    # 先Norm再MLP
        return x
```

#### GPT (完整模型)
```python
class GPT(nn.Module):
    def forward(self, tokens):
        pos_emb = self.transformer.wpe(positions)
        tok_emb = self.transformer.wte(tokens)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
```

---

## 4. 关键训练技巧

### 4.1 权重共享 (Weight Tying) ⭐

```python
self.transformer.wte.weight = self.lm_head.weight  # 共享内存！
```

- Token 嵌入矩阵与输出头**共享同一权重**
- **节省 ~38M 参数** (50257 × 768)
- 同时作为正则化提升性能

### 4.2 残差流缩放初始化

```python
def _init_weights(self, module):
    std = 0.02
    if hasattr(module, 'NANOGPT_SCALE_INIT'):
        std *= (2 * self.config.n_layer) ** -0.5  # 缩小残差投影
```

- 标准差 0.02 用于大多数参数
- 输出投影 (`c_proj`) 按 $1/\sqrt{2 \times n\_layer}$ 缩放
- 防止残差流随深度增长过大

### 4.3 Vocab Size 对齐 (50257 → 50304)

填充词表到 128 的倍数，提升 GPU tensor core 利用率。

---

## 5. Flash Attention ⭐

### 5.1 标准 Attention 的问题

```python
# 标准: O(T²) 内存，需要实体化 T×T 矩阵
att = (q @ k.T) * (1/sqrt(d_k))
att = att.masked_fill(mask == 0, float('-inf'))
att = F.softmax(att, dim=-1)
y = att @ v
```

### 5.2 Flash Attention 优势

```python
# Flash Attention: 一行替换
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

- **内核融合**：Q×K^T、masking、softmax、×V 合为一个 GPU 内核
- **IO感知**：从不在 HBM 中实体化完整 T×T 注意力矩阵
- 使用 tiling 在 SRAM（片上内存）中计算
- 长序列显著加速

---

## 6. 完整预训练配置 (FineWeb-Edu 10B)

### 6.1 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 总 batch size | 524,288 tokens (~0.5M = 2^19) | |
| Micro batch size (B) | 64 | |
| 序列长度 (T) | 1024 | |
| 梯度累积步数 | total/(B×T×GPU数) | |
| 最大学习率 | 6e-4 | |
| 最小学习率 | 6e-5 (最大的10%) | |
| Warmup 步数 | 715 | |
| 最大训练步数 | 19,073 (~1 epoch for 10B tokens) | |
| Weight decay | 0.1 | |
| Adam betas | (0.9, 0.95) | |
| 梯度裁剪 | 1.0 (max norm) | |

### 6.2 学习率调度 (Cosine with Warmup)

```python
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps           # 线性 warmup
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 余弦衰减
    return min_lr + coeff * (max_lr - min_lr)
```

### 6.3 Optimizer 配置

```python
# 分离衰减/不衰减参数
decay_params = [p for p in params if p.dim() >= 2]      # 权重矩阵
nodecay_params = [p for p in params if p.dim() < 2]     # 偏置、LayerNorm

optimizer = torch.optim.AdamW(
    [{'params': decay_params, 'weight_decay': 0.1},
     {'params': nodecay_params, 'weight_decay': 0.0}],
    lr=learning_rate, betas=(0.9, 0.95), fused=True
)
```

### 6.4 训练优化清单

1. **Mixed Precision (bfloat16)**: `torch.autocast(dtype=torch.bfloat16)`
2. **TF32 matmuls**: `torch.set_float32_matmul_precision('high')`
3. **Flash Attention**: 替换手动注意力计算
4. **梯度累积**: 模拟大 batch 在有限 GPU 内存上
5. **DDP 同步优化**: 只在最后一个 micro-step 同步梯度
6. **Fused AdamW**: 单 CUDA 内核完成所有优化器操作
7. **torch.compile** (可选): JIT 编译进一步加速

### 6.5 分布式训练 (DDP)

```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```

```python
# 只在最后 micro-step 同步梯度
model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
```

---

## 7. HellaSwag 评测

### 7.1 任务描述

常识推理基准：从4个选项中选出最合理的句子续写。

### 7.2 评测方法

```python
def get_most_likely_row(tokens, mask, logits):
    # 计算每个选项的交叉熵损失（只在续写部分）
    # 选择平均损失最低的选项
    avg_loss = masked_shift_losses.sum(dim=1) / shift_mask.sum(dim=1)
    pred = avg_loss.argmin().item()
```

### 7.3 基准结果

| 模型 | HellaSwag 准确率 |
|------|----------------|
| GPT-2 (124M) | 29.55% |
| GPT-2-XL (1558M) | 48.93% |

---

## 8. 从验证到全流程

### 8.1 单 batch 过拟合验证

```
Loss: 10.96 → 6.61 → 4.37 → ... → 0.01  (50步)
```

**关键洞察**：成功过拟合单个 batch = 模型架构正确，可以学习。

### 8.2 随机模型 vs 预训练模型

- 随机模型初始 loss ≈ $\log(50257) \approx 10.82$（均匀分布）
- 预训练模型 top-50 概率之和 ≈ 0.9 → 分布集中
- 随机模型 top-50 概率之和 ≈ 0.006 → 均匀扩散

---

## 核心要点总结

1. **GPT = Decoder-only Transformer + Pre-Norm + GELU + 学习型位置编码**
2. **权重共享**节省 38M 参数并提升性能
3. **Flash Attention** 是 drop-in 加速（不改变数学结果，只改变计算方式）
4. **梯度累积**让有限 GPU 内存可以模拟大 batch（0.5M tokens）
5. **Cosine LR + Warmup** 是现代训练的标准实践
6. **bfloat16 > float16**：相同动态范围，无需 loss scaling
7. **评测三件套**：验证集 loss + HellaSwag + 文本生成
8. **Vocab padding (128倍数)** 提升 GPU 利用率
