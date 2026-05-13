# 项目上下文总览：Attention Residuals 到 AttnComm-QMIX-L2

> 用途：下次新对话开始时，直接读取本文件，让 AI 快速恢复项目背景、关键决策、代码状态、实验结果、同步路径和论文写作口径。

更新时间：2026-05-13  
本地工作目录：`D:\MARL\MRAL-Server\MARL\pymarl`  
服务器对应目录：`/home/xhl009/MARL/pymarl`  
服务器结果目录：`/home/xhl009/MARL/pymarl-results`  
本地结果目录：`D:\MARL\MRAL-Server\MARL\pymarl-results`

## 1. 当前论文主线

当前研究主题是：

**Attention Residuals 从大语言模型到多智能体强化学习的迁移适配性研究，并进一步验证 agent-wise attention communication 是否更符合 MARL 协同瓶颈。**

现在论文主线已经从“只说直接迁移不稳定”调整为：

```text
Direct depth-wise Attention Residuals transfer is unstable and costly.
However, moving selective attention from residual-depth aggregation to
agent-wise communication gives clearer positive signals on 5m_vs_6m.
The issue is not whether attention is useful, but where the transferred
structure is placed relative to MARL task bottlenecks.
```

中文口径：

- 不把 `qmix_attnres` 或 `qmix_attncomm_l2_*` 写成 SOTA 或稳定强算法。
- 原结论仍成立：直接把 LLM 的 depth-wise Attention Residuals 硬接到浅层 recurrent MARL agent 后面，不稳定且成本高。
- AttnComm 不是推翻原结论，而是进一步解释原结论：attention 思想本身不是问题，迁移位置更关键。
- 论文后半部分可以把 AttnComm 作为重要亮点：`agent-wise communication transfer` 比 `depth-wise residual transfer` 更贴近 SMAC 的 partial observability 和 coordination bottleneck。
- 当前 AttnComm 结果只能写成 `preliminary positive signal`，不能写成跨地图稳定提升。
- 当前论文数据已经可以收口；不再强制补跑 `3s5z`、`5m_vs_6m` seed4/5、新 hard map，或把 AttnComm 扩到 IQL/VDN。

最终收束句可用：

```text
Directly transferring depth-wise Attention Residuals to shallow recurrent
MARL agents is unstable and costly. However, when the same selective-attention
idea is moved from residual-depth aggregation to agent-wise communication,
AttnComm-QMIX-L2 shows clearer positive signals on the diagnostic 5m_vs_6m map.
This suggests that the key issue is not whether attention is useful, but where
the transferred structure is placed relative to MARL task bottlenecks.
```

## 2. 方法边界和关键决策

保持不变的部分：

- 不改 SC2/SMAC 环境源码。
- 不改 learner。
- 不改 mixer。
- 不改 action selector。
- 不改 QMIX 的 mixing network。
- 第一版 AttnComm 只做 QMIX，不扩到 VDN/IQL。
- 不做 multi-head、4 层重型通信或大规模调参。

只修改 agent 表示路径：

```text
GRU hidden -> agent-wise communication representation -> individual Q -> QMIX
```

AttnComm-QMIX-L2 的稳定性设计：

- 默认 2 层 communication。
- single-head attention。
- hidden dim 64。
- dropout 0.0。
- q/k 使用 normalized hidden。
- value 使用原 hidden。
- gated residual fusion。
- gate bias 初始化为 `-2.0`。
- recurrent hidden state 返回原始 GRU hidden，不把通信后的 hidden 反馈到下一时间步。

两种通信源：

```text
qmix_attncomm_l2_other:
  N_i = {j | j != i}
  强制 message 来自其他智能体。

qmix_attncomm_l2_self:
  N_i = {1, ..., n}
  允许同时选择自身和其他智能体。
```

当前结果支持的解释：

- `other-only` 比 `self-inclusive` 更强。
- 这更支持“跨智能体通信”解释，而不是简单的 self-attention 表示变换。

## 3. 已完成代码工作

### 3.1 AttnRes 直接迁移模块

已有结构：

```text
obs / last action / agent id
-> fc1
-> GRUCell
-> DepthwiseAttentionResidual
-> Q head
-> QMIX / VDN / IQL backend
```

主要文件：

```text
src/modules/agents/attn_residual.py
src/modules/agents/attnres_rnn_agent.py
src/modules/agents/depth_mlp_rnn_agent.py
src/modules/agents/__init__.py
```

已完成：

- `DepthwiseAttentionResidual`
  - 支持 `full` 和 `block`。
  - 使用 RMSNorm。
  - zero-initialized learned pseudo-queries。
  - 支持 attention weight logging。
- `AttnResRNNAgent`
  - `fc1 -> GRUCell -> DepthwiseAttentionResidual -> fc2`
- `DepthMLPRNNAgent`
  - depth-only control。
- `BasicMAC.get_and_reset_attention_weight_stats()`
  - 将 agent 内部 attention stats 转发给 learner。
- `QLearner.train()`
  - 在 logger interval 记录 attention stats。

### 3.2 AttnComm-QMIX-L2 模块

新增稳定版 agent-wise communication：

```text
inputs
-> fc1
-> GRUCell
-> reshape [batch, n_agents, hidden]
-> AgentAttentionCommunication
-> fc2
-> individual Q
-> QMIX
```

新增文件：

```text
src/modules/agents/attn_comm.py
src/modules/agents/attncomm_rnn_agent.py
```

修改文件：

```text
src/modules/agents/__init__.py
src/config/default.yaml
tests/test_attn_res_rnn_agent.py
src/controllers/basic_controller.py
src/learners/q_learner.py
```

核心公式：

```text
q_i = W_q norm(h_i)
k_j = W_k norm(h_j)
v_j = W_v h_j
alpha_ij = softmax_j(q_i^T k_j / sqrt(d))
m_i = sum_j alpha_ij v_j
g_i = sigmoid(W_g [h_i, m_i])
h_i' = h_i + g_i * W_o m_i
```

Sacred logging key：

```text
attn_comm_l{layer}_to{target}_from{source}
```

Agent 注册：

```python
REGISTRY["attncomm_rnn"] = AttnCommRNNAgent
```

### 3.3 AttnComm 配置

新增配置：

```text
src/config/algs/qmix_attncomm_l2_other.yaml
src/config/algs/qmix_attncomm_l2_self.yaml
```

共同设置：

```yaml
agent: "attncomm_rnn"
learner: "q_learner"
double_q: True
mixer: "qmix"
rnn_hidden_dim: 64
attn_comm_enabled: True
attn_comm_layers: 2
attn_comm_hidden_dim: 64
attn_comm_dropout: 0.0
record_comm_attn_weights: True
```

差异：

```yaml
qmix_attncomm_l2_other:
  attn_comm_source_mode: "other_only"
  name: "qmix_attncomm_l2_other"

qmix_attncomm_l2_self:
  attn_comm_source_mode: "self_inclusive"
  name: "qmix_attncomm_l2_self"
```

### 3.4 脚本

服务器 4GPU 专用脚本：

```text
scripts/run_attncomm_qmix_l2_4gpu_server.sh
```

重要说明：

```text
空闲 GPU 不是固定事实，需要用户每次根据服务器实时状态给出。
历史上曾使用过 GPU 2,3,4,7，但后续不要默认沿用。
```

因此，下次启动实验前必须先让用户确认当前空闲 GPU 列表，或根据用户新给出的 GPU 修改脚本/命令。

脚本设置：

```text
RESULTS_DIR=/home/xhl009/MARL/pymarl-results
SC2PATH=/home/xhl009/MARL/pymarl/3rdparty/StarCraftII_srv
5m_vs_6m t_max=5000000
3s5z t_max=2050000
test_nepisode=32
```

初始任务策略：

- `5m_vs_6m` 跑完整公平对比：
  - `qmix`
  - `qmix_attnres_l2`
  - `qmix_attncomm_l2_other`
  - `qmix_attncomm_l2_self`
  - seeds `1,2,3`
- `3s5z` 只跑新增 AttnComm：
  - `qmix_attncomm_l2_other`
  - `qmix_attncomm_l2_self`
  - seeds `1,2,3`
- 不重跑 `3s5z qmix` baseline。
- 初始计划不重跑 `3s5z qmix_attnres_l2`。

后续实际补跑状态：

- 为补齐本地 summary 缺口，已额外 rerun `3s5z qmix_attnres_l2` seeds `1,2,3`。
- 已 rerun `3s5z qmix_attncomm_l2_other` seed `3`，覆盖旧 FAILED run。
- 已 rerun `3s5z qmix_attncomm_l2_self` seed `3`，覆盖旧 missing slot。
- 这些 rerun 已同步到本地 Sacred runs `18-22`，并通过 `scripts/finalize_3s5z_rerun_local.ps1` 验证。

结果处理脚本：

```text
scripts/summarize_marl_transfer_adaptation.py
scripts/plot_marl_transfer_curves.py
scripts/plot_comm_attn_heatmaps.py
scripts/finalize_3s5z_rerun_local.ps1
```

已支持：

- `qmix_attncomm_l2_other`
- `qmix_attncomm_l2_self`
- `--cross-pairs baseline:candidate`
- AttnComm paired delta。
- AttnComm curve labels。
- communication attention heatmap。
- 本地 3s5z rerun 收口脚本：重新汇总、画图、验证关键配置 3/3 complete，并可复制 3s5z figures 到论文目录。

### 3.5 测试状态

本地已验证：

```powershell
python -m unittest tests.test_attn_res_rnn_agent
python -m unittest discover tests
python -m py_compile src\modules\agents\attn_comm.py src\modules\agents\attncomm_rnn_agent.py scripts\plot_comm_attn_heatmaps.py scripts\summarize_marl_transfer_adaptation.py scripts\plot_marl_transfer_curves.py
```

结果：

```text
tests.test_attn_res_rnn_agent: 13 tests OK
unittest discover tests: 16 tests OK
py_compile: OK
```

注意：

- Windows 本地无法用 `bash -n` 检查 shell 脚本，因为 WSL `/bin/bash` 不存在。
- 服务器上使用 `DRY_RUN=1 bash scripts/run_attncomm_qmix_l2_4gpu_server.sh` 检查 manifest。

## 4. 当前实验结果

### 4.1 已同步结果文件

用户已将服务器 `5m_vs_6m` 与 `3s5z` rerun 结果同步到本地：

```text
D:\MARL\MRAL-Server\MARL\pymarl-results\diagnostics\marl_transfer_primary_qmix_table.csv
D:\MARL\MRAL-Server\MARL\pymarl-results\diagnostics\marl_transfer_cross_algorithm_aggregate.csv
D:\MARL\MRAL-Server\MARL\pymarl-results\diagnostics\marl_transfer_missing_or_partial.csv
D:\MARL\MRAL-Server\MARL\pymarl-results\figures\
D:\MARL\MRAL-Server\MARL\pymarl-results\sacred\18
D:\MARL\MRAL-Server\MARL\pymarl-results\sacred\19
D:\MARL\MRAL-Server\MARL\pymarl-results\sacred\20
D:\MARL\MRAL-Server\MARL\pymarl-results\sacred\21
D:\MARL\MRAL-Server\MARL\pymarl-results\sacred\22
```

本地收口命令已执行并通过：

```powershell
cd D:\MARL\MRAL-Server\MARL\pymarl
powershell -ExecutionPolicy Bypass -File scripts\finalize_3s5z_rerun_local.ps1 -CopyPaperFigures
```

验证结果：

```text
Validation passed. 3s5z rerun summaries and figures are complete.
```

### 4.2 5m_vs_6m 当前可写结果

`5m_vs_6m`，`t_max=5.0M`，paired seeds `1/2/3`，当前四个关键配置均已 `3/3` complete：

```text
qmix:
  final win 0.5208
  best win  0.8021
  AUC       0.4986
  wall      24.80 h

qmix_attnres_l2:
  final win 0.5833
  best win  0.8021
  AUC       0.5028
  wall      33.00 h
  vs qmix final delta +0.0625
  vs qmix AUC delta   +0.0043
  paired final wins 2/3
  paired AUC wins   2/3
  wall ratio about 1.34x
  interpretation: weak positive mean signal, but AUC gain is small and final-win variance is high

qmix_attncomm_l2_other:
  final win 0.6354
  best win  0.8542
  AUC       0.5429
  wall      33.40 h
  vs qmix final delta +0.1146
  vs qmix AUC delta   +0.0444
  paired final wins 2/3
  paired AUC wins   2/3
  wall ratio about 1.35x

qmix_attncomm_l2_self:
  final win 0.5938
  best win  0.8646
  AUC       0.5207
  wall      33.34 h
  vs qmix final delta +0.0729
  vs qmix AUC delta   +0.0222
  paired final wins 2/3
  paired AUC wins   2/3
  wall ratio about 1.35x
```

当前解释：

- `qmix_attncomm_l2_other` 和 `qmix_attncomm_l2_self` 都相对 QMIX 有正向均值提升。
- `other-only` 更强，支持显式跨智能体通信解释。
- `qmix_attnres_l2` 完整 5M 后并非完全无效，但 AUC 增益很小、best win 与 QMIX 持平、final-win 方差较大，因此仍不支撑 direct depth-wise transfer 稳定有效的强结论。
- 训练成本约为 QMIX 的 `1.35x`，比 heavy Full/Block AttnRes 更可控。
- 该结果可以作为论文后半部分的亮点，但只能写成 `5m_vs_6m` 上的初步正向信号。

### 4.3 3s5z sanity check 当前结果

`3s5z`：

- 已补齐 rerun，同步后的关键 3 个配置均为 `3/3 complete`。
- 新增 Sacred runs：
  - `18`: `3s5z qmix_attnres_l2 seed=1`
  - `19`: `3s5z qmix_attncomm_l2_other seed=3`
  - `20`: `3s5z qmix_attnres_l2 seed=3`
  - `21`: `3s5z qmix_attnres_l2 seed=2`
  - `22`: `3s5z qmix_attncomm_l2_self seed=3`
- 最新结果：

```text
qmix_attnres_l2:
  complete  3/3
  final win 0.9583
  best win  1.0000
  AUC       0.7608
  wall      18.17 h

qmix_attncomm_l2_other:
  complete  3/3
  final win 0.9792
  best win  1.0000
  AUC       0.8078
  wall      15.34 h

qmix_attncomm_l2_self:
  complete  3/3
  final win 0.9271
  best win  1.0000
  AUC       0.7951
  wall      13.88 h
```

- `3s5z` 仍只作为 sanity check：该地图接近饱和，不能写成跨地图稳定提升的强证据。
- 可写口径：AttnComm 在 `3s5z` 没有明显 collapse，other-only 的 AUC 也高于 AttnRes-L2；但主结论仍依赖诊断地图 `5m_vs_6m`。

### 4.4 Heatmap

已生成并复制到论文 figures：

```text
paper/latex/figures/5m_vs_6m_qmix_attncomm_l2_other_l0_attncomm_attention_heatmap.pdf
paper/latex/figures/5m_vs_6m_qmix_attncomm_l2_self_l0_attncomm_attention_heatmap.pdf
paper/latex/figures/3s5z_qmix_attncomm_l2_other_l0_attncomm_attention_heatmap.pdf
paper/latex/figures/3s5z_qmix_attncomm_l2_other_l1_attncomm_attention_heatmap.pdf
paper/latex/figures/3s5z_qmix_attncomm_l2_self_l0_attncomm_attention_heatmap.pdf
paper/latex/figures/3s5z_qmix_attncomm_l2_self_l1_attncomm_attention_heatmap.pdf
```

解释：

- `other_only` 的 self attention 权重为 0。
- `self_inclusive` 允许 self 权重。
- heatmap 适合作为机制观察材料，不作为强因果证明。

## 5. 论文当前状态

LaTeX 工程：

```text
paper/latex
```

核心章节：

```text
paper/latex/sections/abstract.tex
paper/latex/sections/01_introduction.tex
paper/latex/sections/02_related_work.tex
paper/latex/sections/03_environment_baselines.tex
paper/latex/sections/04_method.tex
paper/latex/sections/05_experimental_setup.tex
paper/latex/sections/06_results.tex
paper/latex/sections/07_discussion.tex
paper/latex/sections/08_conclusion.tex
paper/latex/sections/appendix_checklist.tex
```

### 5.1 已完成的论文主线调整

摘要：

- 已加入 AttnComm-QMIX-L2 在 `5m_vs_6m` 上的初步正向信号。
- 已声明本文不是强性能算法主张。

第 1 章：

- 研究问题从三个扩展为四个。
- 第四个研究问题：direct depth-wise transfer 不稳定是否说明 attention 不适合 MARL？
- 贡献结构改为：
  - 系统评估 LLM Attention Residuals 直接迁移。
  - 发现 heavy depth-wise AttnRes 成本高且不稳定，L2 只有有限信号。
  - 基于结构错位分析提出并初步验证 AttnComm-QMIX-L2。

第 2 章：

- 已把 agent-wise communication 从“后续方向”改成“后半部分扩展验证”。
- 避免与当前第 4/6 章 AttnComm 实验发生口径冲突。

第 4 章：

- 保留 AttnRes 为主方法。
- 新增 `AttnComm-QMIX-L2 Extension` 小节。
- 已写清：
  - 输入 `H^t=[h_1^t,\ldots,h_n^t]`
  - q/k/v agent-wise attention
  - gated residual fusion
  - recurrent hidden 返回通信前 GRU hidden
  - `other-only` 与 `self-inclusive` 两种通信源
- 方法差异表已加入：
  - `qmix_attncomm_l2_other`
  - `qmix_attncomm_l2_self`

第 6 章：

- 新增 `Agent-wise Communication Extension` 小节。
- 已加入 5M AttnComm 结果表：

```text
paper/latex/tables/attncomm_5m_extension_results.tex
```

- 已写：
  - `qmix_attncomm_l2_other` final win 从 0.5208 到 0.6354。
  - AUC 从 0.4986 到 0.5429。
  - `qmix_attncomm_l2_self` 也有正向均值提升但较小。
  - paired seeds final/AUC 都是 2/3 优于 QMIX。
  - `other-only > self-inclusive`。
  - `qmix_attnres_l2` 已完成 3/3，final win 有小幅正向均值，但 AUC 增益很小且方差较大，因此仍写成 direct transfer 弱信号/不稳定。
  - `3s5z` AttnComm sanity check 已补齐，但该地图接近饱和，因此只作为补充验证，不作为跨地图稳定提升的强证据。

第 7 章：

- 核心讨论改成 `Placement Matters：通信维度更适配`。
- 已明确：
  - direct depth-wise transfer 不适配浅层 MARL agent。
  - agent-wise selective communication 更贴合 partial observability 和 coordination。
  - AttnComm 初步结果不推翻原结论，而是解释原结论。
  - 当前结果不能写成稳定 SOTA 或普适提升。

第 8 章：

- 结论已改为：
  - direct transfer 不稳定且成本高。
  - AttnComm-QMIX-L2 在 `5m_vs_6m` 上显示更明确正向信号。
  - 更合理方向是把 selective attention 放到 agent-wise communication。
- 未来工作已改为：
  - 在更有区分度的 SMAC 地图上继续验证 AttnComm 信号，而不是依赖接近饱和的 `3s5z` sanity check。
  - 更多地图比较 `other` 与 `self`。
  - 分析 communication attention weights。
  - 增加 seeds 和统计显著性分析。

### 5.2 论文 figure/table 状态

已复制到论文 figures：

```text
paper/latex/figures/5m_vs_6m_qmix_win_curve.pdf
paper/latex/figures/5m_vs_6m_cross_algorithm_win_curve.pdf
paper/latex/figures/5m_vs_6m_wall_time_bar.pdf
paper/latex/figures/5m_vs_6m_qmix_attncomm_l2_other_l0_attncomm_attention_heatmap.pdf
paper/latex/figures/5m_vs_6m_qmix_attncomm_l2_self_l0_attncomm_attention_heatmap.pdf
paper/latex/figures/3s5z_qmix_win_curve.pdf
paper/latex/figures/3s5z_qmix_attncomm_l2_other_l0_attncomm_attention_heatmap.pdf
paper/latex/figures/3s5z_qmix_attncomm_l2_other_l1_attncomm_attention_heatmap.pdf
paper/latex/figures/3s5z_qmix_attncomm_l2_self_l0_attncomm_attention_heatmap.pdf
paper/latex/figures/3s5z_qmix_attncomm_l2_self_l1_attncomm_attention_heatmap.pdf
```

新增或已有 tables：

```text
paper/latex/tables/attncomm_5m_extension_results.tex
paper/latex/tables/model_cost_summary.tex
paper/latex/tables/paired_seed_delta.tex
```

### 5.3 编译状态

最近编译命令：

```powershell
cd D:\MARL\MRAL-Server\MARL\pymarl\paper\latex
C:\Users\Terry396\.codex\plugins\cache\openai-bundled\latex-tectonic\0.1.1\bin\tectonic.exe --outdir . main.tex
```

结果：

```text
main.pdf 生成成功
无 LaTeX error
仅有少量已有 Underfull hbox / package warning
```

也曾用：

```powershell
latexmk -xelatex -synctex=1 -interaction=nonstopmode -file-line-error main.tex
```

成功生成 `main.pdf`。

## 6. 同步路径

### 6.1 本地代码同步到服务器

根目录映射：

```text
D:\MARL\MRAL-Server\MARL\pymarl
-> /home/xhl009/MARL/pymarl
```

关键相对路径：

```text
src/modules/agents/attn_comm.py
src/modules/agents/attncomm_rnn_agent.py
src/modules/agents/__init__.py
src/config/default.yaml
src/config/algs/qmix_attncomm_l2_other.yaml
src/config/algs/qmix_attncomm_l2_self.yaml
tests/test_attn_res_rnn_agent.py
scripts/summarize_marl_transfer_adaptation.py
scripts/plot_marl_transfer_curves.py
scripts/plot_comm_attn_heatmaps.py
scripts/run_attncomm_qmix_l2_4gpu_server.sh
scripts/run_5m_attncomm_seed45_4gpu_server.sh
scripts/finalize_3s5z_rerun_local.ps1
paper/project_context_next_chat.md
```

如果同步论文，也同步：

```text
paper/latex/sections/abstract.tex
paper/latex/sections/01_introduction.tex
paper/latex/sections/02_related_work.tex
paper/latex/sections/04_method.tex
paper/latex/sections/06_results.tex
paper/latex/sections/07_discussion.tex
paper/latex/sections/08_conclusion.tex
paper/latex/tables/attncomm_5m_extension_results.tex
paper/latex/figures/5m_vs_6m_qmix_win_curve.pdf
paper/latex/figures/5m_vs_6m_cross_algorithm_win_curve.pdf
paper/latex/figures/5m_vs_6m_wall_time_bar.pdf
paper/latex/figures/5m_vs_6m_qmix_attncomm_l2_other_l0_attncomm_attention_heatmap.pdf
paper/latex/figures/5m_vs_6m_qmix_attncomm_l2_self_l0_attncomm_attention_heatmap.pdf
paper/latex/figures/3s5z_qmix_win_curve.pdf
paper/latex/figures/3s5z_qmix_attncomm_l2_other_l0_attncomm_attention_heatmap.pdf
paper/latex/figures/3s5z_qmix_attncomm_l2_other_l1_attncomm_attention_heatmap.pdf
paper/latex/figures/3s5z_qmix_attncomm_l2_self_l0_attncomm_attention_heatmap.pdf
paper/latex/figures/3s5z_qmix_attncomm_l2_self_l1_attncomm_attention_heatmap.pdf
```

### 6.2 服务器结果同步回本地

```text
/home/xhl009/MARL/pymarl-results/sacred
-> D:\MARL\MRAL-Server\MARL\pymarl-results\sacred

/home/xhl009/MARL/pymarl-results/launcher_logs
-> D:\MARL\MRAL-Server\MARL\pymarl-results\launcher_logs

/home/xhl009/MARL/pymarl-results/run_manifests
-> D:\MARL\MRAL-Server\MARL\pymarl-results\run_manifests

/home/xhl009/MARL/pymarl-results/diagnostics
-> D:\MARL\MRAL-Server\MARL\pymarl-results\diagnostics

/home/xhl009/MARL/pymarl-results/figures
-> D:\MARL\MRAL-Server\MARL\pymarl-results\figures
```

## 7. 服务器命令

验证代码：

```bash
cd /home/xhl009/MARL/pymarl
conda activate pymarl-sc2
python -m unittest tests.test_attn_res_rnn_agent
python -m py_compile src/modules/agents/attn_comm.py src/modules/agents/attncomm_rnn_agent.py scripts/plot_comm_attn_heatmaps.py
```

Dry-run：

```bash
DRY_RUN=1 bash scripts/run_attncomm_qmix_l2_4gpu_server.sh
```

正式启动：

```bash
bash scripts/run_attncomm_qmix_l2_4gpu_server.sh
```

监控：

```bash
tmux ls
watch -n 5 nvidia-smi
tail -f /home/xhl009/MARL/pymarl-results/launcher_logs/<log_file>.log
```

汇总：

```bash
python scripts/summarize_marl_transfer_adaptation.py \
  --sacred-dir /home/xhl009/MARL/pymarl-results/sacred \
  --output-dir /home/xhl009/MARL/pymarl-results/diagnostics \
  --maps 5m_vs_6m,3s5z \
  --primary-configs qmix,qmix_attnres_l2,qmix_attncomm_l2_other,qmix_attncomm_l2_self \
  --seeds 1,2,3 \
  --include-cross \
  --cross-pairs qmix:qmix_attnres_l2,qmix:qmix_attncomm_l2_other,qmix:qmix_attncomm_l2_self
```

画图：

```bash
python scripts/plot_marl_transfer_curves.py \
  --sacred-dir /home/xhl009/MARL/pymarl-results/sacred \
  --output-dir /home/xhl009/MARL/pymarl-results/figures \
  --seeds 1,2,3

python scripts/plot_comm_attn_heatmaps.py \
  --sacred-dir /home/xhl009/MARL/pymarl-results/sacred \
  --output-dir /home/xhl009/MARL/pymarl-results/figures
```

## 8. 下一步优先级

### 8.1 实验

1. `3s5z` AttnComm sanity check 已完成并同步，本地验证已通过。
2. 当前定稿前不再强制占用 GPU 补跑实验；服务器 GPU 2/3/4/6 即使空闲，也可以先保留。
3. 不再补 `3s5z`：该地图已完成 sanity check，best win 接近饱和，继续补 baseline 或更多 seeds 对主结论增益很小。
4. 不立即补 `5m_vs_6m` seed4/5：它能增强统计可信度，但单个 5M run 成本较高，当前 3 seeds + paired analysis + `3s5z` sanity check 已可支撑谨慎结论。
5. 不立即跑新地图，也不扩 AttnComm 到 IQL/VDN；这些只作为 optional future queue。
6. 每次新增结果后运行：

```powershell
cd D:\MARL\MRAL-Server\MARL\pymarl
powershell -ExecutionPolicy Bypass -File scripts\finalize_3s5z_rerun_local.ps1 -CopyPaperFigures
```

### 8.2 论文

下一次更新论文时，重点检查：

- `3s5z` 只能作为 sanity check：已补齐，但接近饱和。
- `5m_vs_6m` 是主要正向诊断地图。
- 不把 `3s5z` 写成强结论地图。
- 不把 AttnComm 写成 SOTA。
- 不声称跨地图稳定提升。
- 保持“direct transfer 不稳定，communication dimension 更适配”的主线。

### 8.3 如果要继续增强结果

优先级建议：

1. 仅当时间非常宽裕时，补 `5m_vs_6m` seed4/5：`qmix`, `qmix_attnres_l2`, `qmix_attncomm_l2_other`, `qmix_attncomm_l2_self`。
2. 若要补新地图，优先 `3s5z_vs_3s6z`，先跑 `qmix`, `qmix_attnres_l2`, `qmix_attncomm_l2_other` seeds `1,2,3`。
3. 再考虑 `qmix_attncomm_l2_self` 或 `8m_vs_9m`。
4. 暂时不建议继续堆更深 AttnRes、更重 AttnComm，或为了当前论文强行扩到 IQL/VDN。

## 9. 下次对话建议入口

如果继续查结果：

```text
请读取 D:/MARL/MRAL-Server/MARL/pymarl/paper/project_context_next_chat.md，检查本地 pymarl-results 里最新 5m_vs_6m 和 3s5z 结果，并确认 3s5z sanity check 仍只作为补充证据。
```

如果继续同步服务器：

```text
请读取 D:/MARL/MRAL-Server/MARL/pymarl/paper/project_context_next_chat.md，给我需要从服务器 /home/xhl009/MARL/pymarl-results 同步回本地的具体路径和同步后要跑的汇总命令。
```

如果继续写论文：

```text
请读取 D:/MARL/MRAL-Server/MARL/pymarl/paper/project_context_next_chat.md，按照“direct transfer 不稳定，agent-wise communication 更适配”的主线，帮我润色摘要、结果和结论。
```

如果继续跑实验：

```text
请读取 D:/MARL/MRAL-Server/MARL/pymarl/paper/project_context_next_chat.md，根据我实时给出的空闲 GPU 列表检查/修改 AttnComm 服务器脚本，并确认不会重跑 3s5z baseline。
```

## 10. 2026-05-13 Optional 5m_vs_6m Seed4/5 Insurance Queue

当前论文仍不需要等待新增实验才能收口；这组实验只作为 GPU 2/3/4/6 空闲时的后台保险实验。

已新增服务器脚本：

```text
scripts/run_5m_attncomm_seed45_4gpu_server.sh
```

默认任务：

```text
map=5m_vs_6m
t_max=5000000
test_nepisode=32
seeds=4,5

GPU 2: qmix seed4, qmix seed5
GPU 3: qmix_attnres_l2 seed4, qmix_attnres_l2 seed5
GPU 4: qmix_attncomm_l2_other seed4, qmix_attncomm_l2_other seed5
GPU 6: qmix_attncomm_l2_self seed4, qmix_attncomm_l2_self seed5
```

服务器端 dry run：

```bash
cd /home/xhl009/MARL/pymarl
conda activate pymarl-sc2
DRY_RUN=1 bash scripts/run_5m_attncomm_seed45_4gpu_server.sh
```

正式启动：

```bash
cd /home/xhl009/MARL/pymarl
conda activate pymarl-sc2
bash scripts/run_5m_attncomm_seed45_4gpu_server.sh
```

脚本会写入：

```text
/home/xhl009/MARL/pymarl-results/run_manifests/<RUN_ID>.tsv
/home/xhl009/MARL/pymarl-results/launcher_logs/<RUN_ID>/
```

完成后优先汇总 seeds 1-5 的 `5m_vs_6m`：

```bash
python scripts/summarize_marl_transfer_adaptation.py \
  --sacred-dir /home/xhl009/MARL/pymarl-results/sacred \
  --output-dir /home/xhl009/MARL/pymarl-results/diagnostics \
  --maps 5m_vs_6m \
  --primary-configs qmix,qmix_attnres_l2,qmix_attncomm_l2_other,qmix_attncomm_l2_self \
  --seeds 1,2,3,4,5 \
  --include-cross \
  --cross-pairs qmix:qmix_attnres_l2,qmix:qmix_attncomm_l2_other,qmix:qmix_attncomm_l2_self
```

论文使用原则：

```text
不等这组实验定稿。
若趋势一致，再把 AttnComm 5m_vs_6m 表更新为 5 seeds。
若趋势变弱，作为 robustness discussion 或 appendix。
若反转，改写为 promising but seed-sensitive；仍不写 SOTA 或跨地图稳定提升。
```
