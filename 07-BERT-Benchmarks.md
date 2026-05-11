# Lecture 07: BERT 与评测基准

> 核心主题：GLUE/SuperGLUE 基准、BERT vs GPT zero-shot 评测、HLE、ARC-AGI-3

---

## 1. 模型架构分类

| 类型 | 代表 | 预训练目标 | 特点 |
|------|------|-----------|------|
| Encoder-only | BERT | MLM + NSP | 双向上下文，适合分类 |
| Decoder-only | GPT | 下一词预测 | 单向/自回归，适合生成 |
| Encoder-Decoder | T5 | Span Corruption | 适合 seq2seq |

---

## 2. Zero-Shot 评测策略

### 2.1 BERT (Encoder-only) — Fill-Mask Scoring

```
模板: "{sentence} It was [MASK]."
比较: P("great") vs P("terrible") 在 [MASK] 位置
```

**局限**：BERT 被设计为需要微调的，fill-mask 只是权宜之计。

### 2.2 GPT-2 (Decoder-only) — 似然评分

```
计算: log P(label_tokens | prompt)
选择: 分数最高的标签
```

### 2.3 Instruction-tuned (Qwen3) — Chat 格式

```
用 apply_chat_template 格式化
比较: 标签 token 在下一位置的 logits
```

---

## 3. GLUE 基准详解 ⭐

### 3.1 单句任务

| 任务 | 类型 | 指标 | 数据量 | 说明 |
|------|------|------|--------|------|
| **CoLA** | 二分类 | MCC | 8,551 | 语法可接受性判断 |
| **SST-2** | 二分类 | Accuracy | 67,349 | 情感分类 (电影评论) |

### 3.2 相似度/释义任务

| 任务 | 类型 | 指标 | 数据量 | 说明 |
|------|------|------|--------|------|
| **MRPC** | 二分类 | Acc + F1 | 3,668 | 释义检测 |
| **STS-B** | 回归 | Pearson/Spearman | 5,749 | 语义相似度 [1-5] |
| **QQP** | 二分类 | Acc + F1 | 363,846 | 重复问题检测 |

### 3.3 推理任务

| 任务 | 类型 | 指标 | 数据量 | 说明 |
|------|------|------|--------|------|
| **MNLI** | 三分类 | Accuracy | 392,702 | 自然语言推理 (蕴含/中立/矛盾) |
| **QNLI** | 二分类 | Accuracy | 104,743 | 问答蕴含 (基于SQuAD) |
| **RTE** | 二分类 | Accuracy | 2,490 | 文本蕴含识别 |
| **WNLI** | 二分类 | Accuracy | 635 | 指代消解蕴含 |

---

## 4. Zero-Shot 实验结果 ⭐⭐

### 4.1 SST-2 (情感分类)

| 模型 | 准确率 |
|------|--------|
| bert-base (110M) | 0.565 |
| bert-large (340M) | 0.580 |
| gpt2 (117M) | 0.725 |
| Qwen3-0.6B | 0.760 |
| **Qwen3-1.7B** | **0.880** |

### 4.2 MRPC (释义检测)

| 模型 | 准确率 | F1 |
|------|--------|-----|
| bert-base | 0.315 | 0.000 |
| bert-large | 0.315 | 0.000 |
| gpt2 | 0.530 | 0.608 |
| Qwen3-0.6B | 0.695 | 0.818 |
| **Qwen3-1.7B** | **0.745** | **0.827** |

### 4.3 RTE (文本蕴含)

| 模型 | 准确率 |
|------|--------|
| bert-base | 0.460 |
| gpt2 | 0.460 |
| Qwen3-0.6B | 0.690 |
| **Qwen3-1.7B** | **0.815** |

### 4.4 MNLI (三分类推理)

| 模型 | 准确率 | Macro-F1 |
|------|--------|----------|
| bert-base | 0.410 | 0.194 |
| gpt2 | 0.335 | 0.267 |
| Qwen3-0.6B | 0.455 | 0.288 |
| **Qwen3-1.7B** | **0.775** | **0.771** |

---

## 5. 关键发现

### 5.1 BERT 在 Zero-shot 下近乎无用

- SST-2: 仅 56.5%（接近随机 50%）
- MRPC: F1=0.000（全部预测为同一类）
- RTE/MNLI: 接近随机水平

**原因**：BERT 被设计为通过**微调**使用，fill-mask 评分是不恰当的用法。

### 5.2 指令微调 + 规模 = 强 Zero-shot

- Qwen3-1.7B 在所有任务上远超基线
- 组合效应：指令微调 × 规模增大

### 5.3 架构适配性

| 任务类型 | 最适合架构 |
|----------|-----------|
| 分类（带微调） | BERT (Encoder) |
| Zero-shot 生成 | GPT (Decoder) |
| Zero-shot 分类 | Instruction-tuned Decoder |

---

## 6. HLE (Humanity's Last Exam)

### 6.1 基本信息

- **来源**：Center for AI Safety
- **规模**：2,500 道静态问答题
- **领域**：数学(1021)、生物(280)、CS/AI(241)、物理(230)等
- **答案类型**：exactMatch(1909) + multipleChoice(591)
- **多模态**：342/2500 包含图片

### 6.2 难度

- 设计为"人类最后的考试" — PhD/专家级别
- 小模型 (0.5-3B) 得分接近 **0%**
- 即使 GPT-4 也仅约 5-10%

### 6.3 评测格式

```
Explanation: {explanation}
Answer: {answer}
Confidence: {0%-100%}
```

---

## 7. ARC-AGI-3

### 7.1 基本信息

- **类型**：交互式基准（非静态数据集）
- **目标**：测试技能获取效率 — 探索、发现目标、规划
- **动作空间**：7个动作 + RESET

### 7.2 实验结果

| Agent | 得分 |
|-------|------|
| Random (100步) | 0.0 |
| Claude Opus (40步) | 0.0 |
| GPT-4o-mini | 0.0 |

**结论**：即使最强 LLM 也无法在 ARC-AGI-3 上得分 — 真正的推理/探索能力仍是开放问题。

---

## 8. 基准难度层级

```
GLUE (2018)         ← 基本"已解决"，任务相对简单
    ↓
SuperGLUE (2019)    ← 更难的后继
    ↓
HLE (2024)          ← 专家级，小模型~0%
    ↓
ARC-AGI-3 (2024)   ← 交互式，所有模型 0%
```

---

## 9. 评测方法论

### 9.1 指标选择

| 任务特性 | 适用指标 |
|----------|---------|
| 平衡分类 | Accuracy |
| 不平衡分类 | F1, MCC |
| 回归 | Pearson/Spearman |
| 三分类推理 | Accuracy + Macro-F1 |

### 9.2 似然评分实现

```python
def causal_label_score(tokenizer, model, prompt, label_str):
    # 拼接 prompt + label_str
    # 计算 log P(label tokens | prompt)
    # = sum of log-probs for tokens after prompt
    full_ids = tokenizer(prompt + label_str)
    logits = model(full_ids).logits
    log_probs = log_softmax(logits, dim=-1)
    score = sum(log_probs[t, token_id] for t, token_id in response_positions)
    return score
```

---

## 核心要点总结

1. **BERT 无微调几乎无用**：fill-mask 评分不是设计用途，BERT 需要微调
2. **指令微调是 Zero-shot 的关键**：即使 0.6B 模型也能大幅超越 340M BERT
3. **规模 + 指令微调 = 强 Zero-shot**：Qwen3-1.7B 在所有 GLUE 任务上表现优异
4. **评测基准在不断升级**：GLUE → SuperGLUE → HLE → ARC-AGI-3
5. **当前 AI 的真实推理能力有限**：HLE/ARC-AGI-3 仍未解决
6. **评测指标要匹配任务**：不平衡数据用 F1/MCC，不用 Accuracy
