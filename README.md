# REXPERTNET-R2 (DGL) — Query-specific Retrieval + Candidate Reranking (No Diffusion)

本代码库基于 **DMSA** 原始实现（DGL 版）构建：数据预处理、数据加载、训练/测试循环、filtered evaluation 逻辑均保持一致。
在 `model_name=rexpert` 时，使用本文实现的 **REXPERTNET-R2**（R²-DMSA）模型；在 `model_name=dmsa` 时，仍然运行原始 DMSA。

本版本的目标不是“替换 diff_out 但整体与 DMSA 持平”，而是引入 **DMSA 主干难以吞掉的增量通路**：
检索不再只是 logits 的小偏置或中间表示替换，而是进入 **候选集合构造与精排(rerank)**，并通过显式损失训练精排器。

---

## 方法概览（与 DMSA 的关键区别）

DMSA 主干：时序图编码（CompGCN + GRU）得到 `current_h`，再通过
- NCE 分支（`calculate_nce_loss`）得到 `preds`
- 判别增强分支 TimeConvTransE 得到 `score_enhanced`
并融合成 `preds_total`，最后用于 filtered ranking。

REXPERTNET-R2 的关键改动：

1. **Query-specific Structure Recall（结构召回）**
   将历史缓冲区 `list_diff_data` 展平成历史事件集合 `hist_events`，对每个 query 通过结构匹配召回候选事件池：
   - 预测对象：匹配 `(s==s*) OR (r==r*)`
   - 预测主体：匹配 `(o==o*) OR (r==r*)`
   只保留每个 query 最近的 `retrieval_pool_size` 条作为证据池。

2. **Evidence Attention Aggregation（证据注意力汇聚）**
   对证据池事件构造 token：`[e_s || e_r || e_o || e_dt]`，并以 query 表示作为检索 query，
   做 Top-K 注意力汇聚得到证据摘要向量 `c`（替代 diffusion 输出）。

3. **Candidate Union + Reranking（候选联合 + 精排）**
   从 DMSA 的判别增强分支取 Top-M 候选（`rerank_topm`），并将检索 Top-K 事件对应的实体加入候选集合，
   构成固定预算的候选集合（`rerank_max_cands`）。
   精排器对每个候选实体计算增量得分：
   `Δ(o) = MLP([e_o || q || c])`，
   并将 `Δ` 回散到全实体 logits 上用于最终 ranking。
   训练时对候选集合显式加入 `lambda_rank * L_rank`，使精排器不会被 DMSA 主干“学会忽略”。

> 重要：REXPERTNET-R2 **重写了 calculate_nce_loss**，保证检索 bias 不会在 forward 中被 diffusion/diffusion2 的两次调用覆盖（原先这会导致 bias 方向错配）。

---

## 运行方式

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：DGL/PyTorch/CUDA 的安装需与本机环境匹配。若你已能跑通原始 DMSA，则无需额外改动。

### 2) 训练 + 测试（默认验证集为 test）

```bash
python main.py --config configs/rexpert_icews14s.json
```

### 3) 运行原始 DMSA

```bash
python main.py --model_name dmsa --dataset ICEWS14s
```

---

## 配置文件与关键参数

本仓库提供 `configs/rexpert_icews14s.json`。你可以直接在 JSON 中修改参数。

### 验证控制

- `validate_every`: 每多少个 epoch 做一次验证（默认 `1`）
- `validate_split`: 验证使用哪个 split（`test` 或 `valid`），默认 `test`

### 检索相关参数（REXPERTNET-R2）

- `retrieval_pool_size`: 每个 query 结构召回的证据池大小（建议 128~512）
- `retrieval_topk`: 在证据池内做注意力汇聚时的 Top-K（建议 4~16）
- `history_pool_max`: 展平历史缓冲区时最多保留多少条历史事件（过大将增大检索开销）
- `retrieval_time_lambda`: 对注意力 logits 加入时间衰减项的系数（`0` 表示不启用）
- `retrieval_beta`: 可选的“检索实体 bias”强度（加到 NCE logits 上）；默认 `0`（只用 rerank）

> 兼容：`max_hist_steps` 作为旧参数仍可在 config 中出现，会自动映射到 `retrieval_pool_size`。

### 精排相关参数（REXPERTNET-R2）

- `rerank_topm`: 从 DMSA 增强分支取 Top-M 候选加入候选集（建议 128~512）
- `rerank_max_cands`: 最终候选集合预算（建议 256~1024）
- `rerank_eta`: 精排增量分数强度（建议 0.5~2.0）
- `lambda_rank`: 精排候选集合 CE loss 的权重（建议 0.1~1.0）
- `lambda_ret`: 可选的检索对比学习 loss 权重（默认 0）
- `ret_tau`: 对比学习温度系数

---

## 输出与日志

训练过程中会在 `save_dir`（默认 `./logs/`）生成训练日志与模型 checkpoint（若原 DMSA 代码中启用保存）。

---

## 代码结构

- `main.py`: 参数解析、配置加载、训练与测试入口
- `utils.py`: 数据 batch 构建、filtered ranking 评估
- `model.py`: 原始 DMSA 的 NET 主干（保持不变）
- `model_rexpert.py`: REXPERTNET-R2（检索 + 候选精排）实现

