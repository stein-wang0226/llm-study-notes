# Lecture 08: RLHF — SFT, Reward Model, PPO

> 核心主题：InstructGPT 三阶段流水线、监督微调、奖励模型、PPO 对齐
> 参考论文：InstructGPT (Ouyang et al., OpenAI, 2022)

---

## 1. InstructGPT 三阶段流水线 ⭐⭐

```
┌─────────────────────────────────────────────────────┐
│  STAGE 1 — Supervised Fine-Tuning (SFT)              │
│  预训练 GPT → [人类示范数据, 8 epochs] → SFT 模型     │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2 — Reward Model (RM)                         │
│  SFT 模型 → [人类偏好对, 1 epoch] → RM (标量 r)       │
│              Bradley-Terry 目标                       │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 3 — RLHF via PPO                             │
│  SFT 策略 + RM 奖励 + KL 惩罚 → 对齐模型            │
└─────────────────────────────────────────────────────┘
```

**核心思想**：
- 预训练学会**如何**生成流畅文本
- SFT 教会**什么格式**回答
- RM 捕获**人类偏好**
- PPO 直接优化人类偏好，KL 惩罚防止奖励黑客

---

## 2. Stage 1: Supervised Fine-Tuning (SFT)

### 2.1 SFT Loss — 仅响应 Token

$$\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{response}} \log p_\theta(x_t \mid x_{<t})$$

**关键**：Prompt token 标签设为 -100（PyTorch 的 `CrossEntropyLoss` 自动忽略），**只有响应 token 贡献损失**。

### 2.2 Prompt 模板 (Alpaca 格式)

```
### Instruction:
{instruction}

### Input:
{input_text}     (可选)

### Response:
{output}
```

**重要**：SFT、RM、PPO、推理必须使用**相同模板**。

### 2.3 常用 SFT 数据集

| 数据集 | 规模 | 来源 |
|--------|------|------|
| Stanford Alpaca | 52K | GPT-3 self-instruct |
| ShareGPT | ~90K | 真实 ChatGPT 对话 |
| OpenAssistant | 161K | 人类标注 |
| UltraChat | 1.5M | GPT-3.5 生成 |
| MetaMath | 395K | GPT-4 增强 |

### 2.4 实现要点

```python
class SFTDataset:
    def __getitem__(self, idx):
        # 1. 格式化为 Alpaca 模板
        # 2. Tokenize (prompt + response + EOS)
        # 3. labels = input_ids.clone()
        # 4. labels[:prompt_length] = -100   # 掩码 prompt
        return {"input_ids": ..., "labels": ..., "attention_mask": ...}
```

### 2.5 训练结果

| Epoch | Loss | PPL |
|-------|------|-----|
| 1 | 2.22 | 9.21 |
| 2 | 0.26 | 1.30 |
| 8 | 0.23 | 1.26 |

**观察**：大部分学习在第1个 epoch 完成。

---

## 3. Stage 2: Reward Model (RM)

### 3.1 Bradley-Terry 损失 ⭐

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \sigma\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right]$$

- $y_w$：偏好（chosen）响应
- $y_l$：被拒绝（rejected）响应
- $\sigma$：sigmoid 函数

**直觉**：推高 chosen 的分数，压低 rejected 的分数。Sigmoid 使之有概率解释。

### 3.2 RM 架构

```python
class RewardModel(nn.Module):
    def __init__(self):
        self.backbone = GPT2Model.from_pretrained("gpt2")  # 768维
        self.value_head = nn.Linear(768, 1, bias=False)     # 标量输出
    
    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids).last_hidden_state
        # 取最后一个非 padding token 的隐藏状态
        last_pos = attention_mask.sum(1) - 1
        pooled = hidden[range(B), last_pos]    # (B, 768)
        reward = self.value_head(pooled)       # (B, 1) → (B,)
        return reward.squeeze(-1)
```

### 3.3 关键设计

- **从 SFT 权重初始化** backbone（已理解语言和指令格式）
- **只训练 1 个 epoch**（防止在稀缺偏好数据上过拟合）
- **无偏置** value head

### 3.4 常用偏好数据集

| 数据集 | 对数 | 说明 |
|--------|------|------|
| Anthropic HH-RLHF | 170K | 有用性 + 无害性 |
| OpenAI WebGPT | 19K | 搜索辅助对比 |
| Stanford SHP | 385K | StackExchange 投票 |
| UltraFeedback | 64K | GPT-4 评分 |

---

## 4. Stage 3: RLHF via PPO ⭐⭐

### 4.1 RLHF 目标函数

$$\text{objective}(\phi) = \mathbb{E}_{(x,y) \sim \pi_\phi^{RL}}\left[r_\theta(x,y) - \beta \log\frac{\pi_\phi^{RL}(y|x)}{\pi^{SFT}(y|x)}\right]$$

| 项 | 含义 |
|-----|------|
| $r_\theta(x,y)$ | RM 奖励分数 |
| $\beta$ | KL 惩罚系数 (通常 0.02-0.2) |
| $\pi_\phi^{RL}$ | 正在优化的 RL 策略 |
| $\pi^{SFT}$ | 冻结的参考策略 (SFT 模型) |
| KL 项 | 防止策略偏离太远（奖励黑客） |

### 4.2 PPO Clipped Surrogate 目标

$$L^{CLIP} = -\mathbb{E}\left[\min\left(r_t \hat{A}_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \hat{A}_t\right)\right]$$

其中：
- $r_t = \frac{\pi_\phi(a|s)}{\pi_{\phi_{old}}(a|s)}$ — 策略比率
- $\hat{A}_t$ — 优势估计 (GAE)
- $\varepsilon = 0.2$ — 裁剪范围

**作用**：限制策略更新幅度，防止单步过大变化。

### 4.3 ActorCritic 架构

```python
class ActorCritic(nn.Module):
    """
    Actor:  GPT-2 LM head (策略 π_φ，生成文本)
    Critic: 额外线性头 (价值函数 V_φ)
    共享 GPT-2 backbone
    """
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained(sft_path)
        self.value_head = nn.Linear(768, 1)
    
    def get_token_logprobs(self, input_ids):
        # 返回每个 token 的 log p(token | context)
        
    def get_values(self, input_ids):
        # 返回每个序列的价值估计 V(s)
```

### 4.4 PPO 循环

```python
for step in range(ppo_steps):
    # === ROLLOUT (no grad) ===
    prompts = sample_prompts(pool, batch_size)
    responses = actor.generate(prompts, max_new_tokens=64)
    
    # 计算奖励
    rm_scores = reward_model(prompts + responses)
    kl = actor_logprobs - ref_logprobs   # per-token KL
    total_reward = rm_score - beta * kl.sum()
    
    # 计算优势
    values = critic(prompts + responses)
    advantages = total_reward - values   # 简化 GAE
    advantages = (advantages - mean) / std  # 归一化
    
    # === UPDATE (K epochs) ===
    for epoch in range(K):
        new_logprobs = actor.get_logprobs(responses)
        ratio = exp(new_logprobs - old_logprobs)
        
        surr1 = ratio * advantages
        surr2 = clip(ratio, 1-ε, 1+ε) * advantages
        actor_loss = -min(surr1, surr2).mean()
        
        critic_loss = MSE(new_values, returns)
        loss = actor_loss + c_vf * critic_loss
        
        loss.backward()
        optimizer.step()
```

### 4.5 KL 惩罚的作用

**防止奖励黑客 (Reward Hacking)**：

没有 KL 惩罚 → 策略可能发现 RM 的盲点，产出高分但低质的输出
有 KL 惩罚 → 策略被约束在 SFT 分布附近，确保输出质量

### 4.6 训练结果

| Step | Total Reward | RM Score | KL | Actor Loss |
|------|-------------|----------|-----|-----------|
| 1 | -1.923 | -1.923 | 0.000 | 0.208 |
| 50 | -2.232 | -2.168 | 0.642 | 0.162 |
| 100 | -2.944 | -2.702 | 2.424 | 1.059 |

**观察**：KL 随训练增长（策略在偏离 SFT），total_reward = RM_score - 0.1×KL

---

## 5. 完整超参数表

| 参数 | SFT | RM | PPO |
|------|-----|-----|-----|
| 数据集 | Alpaca (5K) | HH-RLHF (2K) | HH-RLHF prompts (500) |
| Epochs | 8 | 1 | 100 steps × 4 inner |
| Batch size | 16 | 16 | 8 |
| Learning rate | 9.65e-6 | 9e-6 | 9.65e-6 |
| Max length | 512 | 512 | prompt + 64 new tokens |
| β (KL coeff) | — | — | 0.1 |
| ε (clip) | — | — | 0.2 |
| γ (discount) | — | — | 1.0 |
| λ (GAE) | — | — | 0.95 |

---

## 6. 定性对比

**Prompt: "Explain what a neural network is to a 10-year-old."**

| 模型 | 输出 |
|------|------|
| Base GPT-2 | 不跟随指令，输出无关文本 |
| SFT | 连贯定义："A transformer is a type of AI system..." |
| PPO | 类似 SFT，略更结构化 |

---

## 7. 关键设计选择

| 选择 | 原因 |
|------|------|
| Response-only SFT loss | 只学输出质量，不"记忆"问题 |
| RM 从 SFT 初始化 | backbone 已理解语言，只需学偏好排序 |
| RM 只训练 1 epoch | 偏好数据稀缺，多 epoch 过拟合 |
| KL 惩罚 β=0.1 | 防止奖励黑客又不过度约束 |
| 冻结参考策略 | 提供稳定的 KL 计算基准 |
| Actor-Critic 共享 backbone | 参数高效（但耦合表示） |

---

## 8. 已知问题与局限

| 问题 | 描述 |
|------|------|
| **奖励黑客** | 策略利用 RM 盲点，高分但低质 |
| **RM 分布偏移** | RM 在 SFT 输出上训练，测试在 PPO 输出上 |
| **标注成本** | InstructGPT 用了 40 个标注员（极贵） |
| **规模限制** | 真实 InstructGPT 用 175B，124M 只演示机制 |

---

## 9. 数学公式速查

### SFT Loss
$$\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{response}} \log p_\theta(x_t \mid x_{<t})$$

### Bradley-Terry RM Loss
$$\mathcal{L}_{\text{RM}} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

### RLHF Objective
$$\text{obj}(\phi) = \mathbb{E}\left[r_\theta(x,y) - \beta \log\frac{\pi_\phi^{RL}(y|x)}{\pi^{SFT}(y|x)}\right]$$

### PPO Clipped Surrogate
$$L^{CLIP} = -\min\left(\frac{\pi_\phi}{\pi_{old}} \hat{A},\ \text{clip}\left(\frac{\pi_\phi}{\pi_{old}}, 1\pm\varepsilon\right) \hat{A}\right)$$

### Total PPO Loss
$$L = L^{CLIP}_{\text{actor}} + c_{vf} \cdot \text{MSE}(V_\phi, R)$$

---

## 10. 延伸阅读

- InstructGPT: https://arxiv.org/abs/2203.02155
- HuggingFace TRL: https://github.com/huggingface/trl
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
- DPO (Direct Preference Optimization): 无需 RM/PPO 的简化方法

---

## 核心要点总结

1. **三阶段流水线**：预训练→SFT→RLHF 是对齐 LLM 的标准范式
2. **SFT 关键**：response-only loss masking + 统一 prompt 模板
3. **RM 关键**：Bradley-Terry 偏好学习，从 SFT 初始化，只训 1 epoch
4. **PPO 关键**：clipped surrogate + KL 惩罚 + frozen 参考策略
5. **KL 惩罚**是防止奖励黑客的核心机制
6. **规模决定质量**：124M 只能演示机制，需 10B+ 才有实际效果
