# 图表、结果与缺失材料清单

本文档用于配合 `chinese_thesis_framework.md`，列出论文初稿需要生成的图表、可直接使用的数据文件，以及当前仍缺失的实验或文献材料。

## 1. 可直接使用的数据文件

结果目录：

```text
D:\MARL\MRAL-Server\MARL\pymarl-results\diagnostics
```

核心文件：

| File | 用途 | 当前状态 |
|---|---|---|
| `marl_transfer_primary_qmix_table.csv` | QMIX 主实验表，包含 final win、best win、AUC、return、wall time | 可直接使用 |
| `marl_transfer_cross_algorithm_pairs.csv` | IQL/QMIX paired seed 对照；VDN candidate 有结果但 baseline 缺失 | 可用于 seed 分析 |
| `marl_transfer_cross_algorithm_aggregate.csv` | 跨算法均值对照 | 可直接使用，但不含 VDN aggregate |
| `marl_transfer_missing_or_partial.csv` | 缺失实验清单 | 当前只缺 VDN baseline |

## 2. 必做表格

### 表 1：SMAC 地图信息

建议列：

```text
Map | Ally units | Enemy units | Unit type | Task characteristic | Role in paper
```

推荐内容：

```text
5m_vs_6m | 5 Marines | 6 Marines | homogeneous | coordination/kiting under disadvantage | main diagnostic map
3s5z | 3 Stalkers + 5 Zealots | same/similar enemy setup | heterogeneous | mixed-unit cooperation | secondary map, partly saturated
```

状态：需要确认官方 map registry 中的精确 enemy composition 与 difficulty 标签。

### 表 2：主要超参数表

建议列：

```text
Hyperparameter | Value | Notes
```

可填内容：

```text
lr | 0.0005 | default PyMARL-style training
batch_size | 32 | replay batch
buffer_size | 5000 | episode buffer
epsilon_start | 1.0 | exploration
epsilon_finish | 0.05 | final epsilon
epsilon_anneal_time | 50000 | annealing schedule
target_update_interval | 200 | target network update
mixing_embed_dim | 32 | QMIX mixer
hypernet_layers | 2 | QMIX hypernetwork
hypernet_embed | 64 | QMIX hypernetwork hidden size
t_max | 2050000 | training timesteps
test_nepisode | 32 | evaluation episodes
seeds | 1,2,3 | repeated runs
```

状态：可直接从配置和 Sacred config 支撑。

### 表 3：模型变体表

建议列：

```text
Variant | Base algorithm | Agent modification | attn_res_layers | mode | Purpose
```

必须包含：

```text
qmix | QMIX | original RNN agent | - | - | baseline
qmix_attnres | QMIX | GRU -> Full AttnRes -> Q | 4 | full | heavy direct transfer
qmix_attnres_l2 | QMIX | GRU -> Full AttnRes -> Q | 2 | full | lightweight transfer
qmix_attnres_block | QMIX | GRU -> Block AttnRes -> Q | 4 | block | block-wise transfer
qmix_depth_mlp | QMIX | GRU -> residual MLP stack -> Q | 4 | no attention | depth-only control
iql_attnres_l2 | IQL | lightweight AttnRes agent | 2 | full | cross-alg validation
vdn_attnres_l2 | VDN | lightweight AttnRes agent | 2 | full | pending baseline comparison
```

状态：可直接使用。

### 表 4：QMIX 主实验结果

来源：`marl_transfer_primary_qmix_table.csv`

建议列：

```text
Map | Variant | Seeds | Final Win | Best Win | Win AUC | Final Return | Wall Hours
```

当前解释重点：

- `5m_vs_6m` 是主要诊断地图。
- `qmix_attnres_l2` 的 AUC 和 best win 最有信号。
- Full/Block 训练成本高且 final win 不稳定。
- `qmix_depth_mlp` 未超过 baseline，说明不是单纯加深即可提升。
- `3s5z` 接近饱和，不宜作为强结论。

状态：可直接使用。

### 表 5：跨算法轻量 AttnRes-L2 结果

来源：`marl_transfer_cross_algorithm_aggregate.csv`

建议列：

```text
Map | Baseline | Candidate | Paired Seeds | Final Win Delta | Best Win Delta | AUC Delta | Wall Time Ratio | Interpretation
```

当前可写：

```text
IQL -> IQL+AttnRes-L2: final +0.0833, best +0.1458, AUC +0.0455
QMIX -> QMIX+AttnRes-L2: final -0.0208, best +0.0208, AUC +0.0567
```

状态：IQL/QMIX 可直接使用；VDN 暂不写入 aggregate 表，除非补齐 baseline。

## 3. 必做图

### 图 1：研究动机图

内容：

```text
Deep LLM residual stream
-> Attention Residuals for depth-wise selection

Shallow MARL recurrent agent
-> direct depth-wise transfer may mismatch
-> agent-wise communication attention is more natural
```

用途：引言。

状态：需要绘制。

### 图 2：原始 RNN Agent 与 AttnRes-RNN Agent 结构对比

内容：

```text
Original:
obs -> fc1 -> GRUCell -> Q head

AttnRes:
obs -> fc1 -> GRUCell -> DepthwiseAttentionResidual -> Q head

Depth-only:
obs -> fc1 -> GRUCell -> residual MLP stack -> Q head
```

用途：方法章节。

状态：需要绘制。

### 图 3：DepthwiseAttentionResidual 模块示意图

内容：

```text
h_0 -> virtual source stack
learnable query -> softmax over depth sources
weighted aggregation -> MLP transform -> residual update
```

用途：方法章节。

状态：需要根据代码绘制；若写公式，需同步检查代码变量命名。

### 图 4：后续 AttnComm 方向概念图

内容：

```text
h_1, h_2, ..., h_n
-> agent-wise attention
-> gated residual fusion
-> individual Q heads
-> QMIX/VDN/IQL
```

用途：讨论或未来工作。

状态：需要绘制。

### 图 5：`5m_vs_6m` QMIX variants win-rate learning curves

曲线：

```text
qmix
qmix_attnres
qmix_attnres_l2
qmix_attnres_block
qmix_depth_mlp
```

建议展示 mean ± std 或 mean with individual seed light lines。

用途：结果分析主图。

状态：需要从 Sacred `info.json` 生成。

### 图 6：`3s5z` QMIX variants win-rate learning curves

同图 5。

用途：说明 `3s5z` 接近饱和，区分度较低。

状态：需要生成。

### 图 7：训练时间柱状图

建议至少画 `5m_vs_6m`：

```text
qmix: 6.59 h
qmix_attnres: 15.14 h
qmix_attnres_l2: 8.29 h
qmix_attnres_block: 13.97 h
qmix_depth_mlp: 8.40 h
```

用途：支撑“heavy variants 成本过高”。

状态：可直接从 CSV 生成。

### 图 8：跨算法 paired delta 图

建议画：

```text
IQL final/best/AUC delta
QMIX final/best/AUC delta
```

或 seed-level paired bars。

用途：展示 AttnRes-L2 在 IQL 上更正向，在 QMIX 上混合。

状态：可直接从 pair/aggregate CSV 生成。

## 4. 当前缺失实验

当前缺失项来自 `marl_transfer_missing_or_partial.csv`：

```text
5m_vs_6m vdn seed1 missing
5m_vs_6m vdn seed2 missing
5m_vs_6m vdn seed3 missing
```

建议处理方式：

1. 如果时间允许，优先补齐 VDN baseline 三个 seed。
2. 如果不补，论文中不要写“VDN 对照证明”，只写“VDN+AttnRes-L2 已运行但由于 baseline 缺失未纳入统计比较”。
3. 若论文篇幅有限，跨算法部分保留 IQL/QMIX 已足够支撑“跨算法初步验证”。

## 5. 需要补充的统计分析

建议至少补三类：

1. Mean/std：每个 map/config 的 final win、best win、AUC 均值和标准差。
2. Paired seed delta：candidate - baseline，尤其 `qmix_attnres_l2` vs `qmix`、`iql_attnres_l2` vs `iql`。
3. 简单显著性检验：3 seeds 很少，不能过度强调 p-value；可报告 paired delta 的方向一致性，例如 final win wins、AUC wins。

当前 summary 已有：

```text
final_win_wins
auc_wins
mean_delta_final_win
mean_delta_best_win
mean_delta_win_auc
mean_wall_time_ratio
```

不足：主表没有 std，需要另写脚本或扩展 summary。

## 6. 写作时应避免的结论

不建议写：

- “AttnRes 显著优于 QMIX。”
- “LLM 结构可直接提升 MARL。”
- “VDN 上也验证有效。”，除非补齐 VDN baseline。

建议写：

- “Heavy AttnRes direct transfer is unstable and costly.”
- “Lightweight AttnRes-L2 shows AUC/best-win signals but does not consistently improve final win rate.”
- “The results indicate an architectural mismatch between depth-wise residual selection in LLMs and shallow recurrent MARL agents.”
- “A more MARL-specific adaptation should use attention for inter-agent communication.”

## 7. 下一步执行顺序

建议按以下顺序推进：

1. 补 VDN baseline 或决定删除 VDN 对照。
2. 生成学习曲线和训练时间图。
3. 扩展 summary 脚本输出 std 和 paired seed delta 表。
4. 补参考文献。
5. 按 `chinese_thesis_framework.md` 逐节扩写：摘要 -> 引言 -> 方法 -> 实验 -> 结果 -> 讨论。
