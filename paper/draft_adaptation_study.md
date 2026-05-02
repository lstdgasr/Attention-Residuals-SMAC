# An Empirical Study of Transferring Attention Residuals from LLMs to Multi-Agent Reinforcement Learning

## Abstract

Attention Residuals were recently proposed to improve information flow in deep language models by replacing fixed residual accumulation with learned depth-wise selection. This paper studies whether the same idea can be transferred to classical multi-agent reinforcement learning (MARL) agents on SMAC. We adapt Attention Residuals to PyMARL by inserting a lightweight depth-wise aggregation module after each recurrent agent hidden state while keeping value factorization unchanged. Across QMIX variants on `5m_vs_6m` and `3s5z`, direct heavy transfer is not consistently beneficial: four-layer full and block variants increase training cost and show unstable final win rates. A two-layer variant is less harmful and improves training AUC and best win rate on `5m_vs_6m`, but the gains do not consistently translate into final performance. These results suggest that residual mechanisms designed for deep LLMs do not directly match the bottlenecks of shallow recurrent MARL agents. We discuss a more MARL-specific follow-up direction that uses attention for inter-agent communication rather than depth-wise residual replacement.

## 1. Introduction

Recent language model architectures increasingly rely on mechanisms that improve information flow across many layers. Attention Residuals (AttnRes) replace fixed residual accumulation with learned attention over previous depth sources, addressing residual dilution in deep PreNorm networks. This motivates a natural question: can such depth-wise residual selection help multi-agent reinforcement learning agents?

Classical SMAC agents in PyMARL, however, are structurally different from LLMs. A standard QMIX agent is shallow:

```text
observation -> MLP -> GRUCell -> action-value head
```

Its main difficulties are partial observability, exploration, credit assignment, and coordination, rather than hundreds of residual layers. Therefore, direct transfer may be mismatched. This work treats that mismatch as the central object of study.

We ask:

1. Does inserting AttnRes after recurrent agent hidden states improve value factorization on SMAC?
2. Is any observed effect due to adaptive depth-wise selection, or simply due to adding more MLP depth?
3. Are lighter AttnRes variants more suitable than heavier LLM-style variants?

Our preliminary results show that direct heavy transfer is unstable, while a two-layer lightweight variant provides limited but non-uniform benefits.

## 2. Background

### 2.1 QMIX and PyMARL Agents

QMIX learns individual agent action-values and mixes them into a monotonic joint action-value conditioned on the global state. In the PyMARL implementation used here, each agent shares a recurrent network:

```text
fc1 -> GRUCell -> fc2
```

The mixer, learner, target updates, double Q-learning, and epsilon-greedy exploration are unchanged in all QMIX variants.

### 2.2 Attention Residuals

Attention Residuals replace uniform residual accumulation with attention over previous depth sources. In an LLM setting, this allows later layers to selectively retrieve earlier representations. The original motivation is residual stream dilution in deep networks.

### 2.3 Transfer Hypothesis

Our initial transfer hypothesis is that depth-wise adaptive aggregation may improve the representation used by each recurrent MARL agent before producing action-values. However, because the base agent is shallow, we expect the benefit to be weaker and less stable than in deep LLMs.

## 3. Method

We evaluate five QMIX-side variants:

| Name | Agent module | Purpose |
|---|---|---|
| `qmix` | original RNN agent | baseline |
| `qmix_attnres` | full AttnRes, 4 layers | direct heavy transfer |
| `qmix_attnres_l2` | full AttnRes, 2 layers | lightweight transfer |
| `qmix_attnres_block` | block AttnRes, 4 layers, block size 2 | block-wise transfer |
| `qmix_depth_mlp` | plain residual MLP depth, no attention | depth-only control |

The direct AttnRes variants use:

```text
observation -> fc1 -> GRUCell -> DepthwiseAttentionResidual -> action-value head
```

The depth-only control uses:

```text
observation -> fc1 -> GRUCell -> residual MLP stack -> action-value head
```

This control is necessary to distinguish the effect of adaptive depth selection from the effect of simply increasing network depth.

## 4. Experimental Setup

Environment: SMAC through PyMARL.

Primary maps:

```text
5m_vs_6m
3s5z
```

Training budget:

```text
t_max = 2,050,000
test_nepisode = 32
seeds = 1, 2, 3
```

Metrics:

- final test win rate
- best test win rate
- normalized win-rate AUC
- final test return
- wall-clock training time

The cross-algorithm validation extends the lightweight AttnRes variant to IQL and VDN on `5m_vs_6m`:

```text
IQL vs IQL+AttnRes-L2
VDN vs VDN+AttnRes-L2
QMIX vs QMIX+AttnRes-L2
```

## 5. Current Results

The current QMIX summary is generated from:

```text
pymarl-results/diagnostics/marl_transfer_primary_qmix_table.csv
```

### 5.1 QMIX Main Results

| Map | Variant | Complete seeds | Final win | Best win | Win AUC | Wall hours |
|---|---:|---:|---:|---:|---:|---:|
| 5m_vs_6m | qmix | 3/3 | 0.6250 | 0.7813 | 0.3926 | 6.59 |
| 5m_vs_6m | qmix_attnres | 3/3 | 0.5417 | 0.7188 | 0.4111 | 15.14 |
| 5m_vs_6m | qmix_attnres_l2 | 3/3 | 0.6042 | 0.8021 | 0.4493 | 8.29 |
| 5m_vs_6m | qmix_attnres_block | 3/3 | 0.5000 | 0.6875 | 0.3951 | 13.97 |
| 5m_vs_6m | qmix_depth_mlp | 1/3 | 0.5313 | 0.7344 | 0.3899 | 5.78 |
| 3s5z | qmix | 2/3 | 0.9167 | 1.0000 | 0.7605 | 8.28 |
| 3s5z | qmix_attnres | 3/3 | 0.9375 | 1.0000 | 0.7504 | 18.29 |
| 3s5z | qmix_attnres_l2 | 3/3 | 0.9583 | 1.0000 | 0.7608 | 10.31 |
| 3s5z | qmix_attnres_block | 2/3 | 0.8281 | 1.0000 | 0.7322 | 16.10 |
| 3s5z | qmix_depth_mlp | 1/3 | 1.0000 | 1.0000 | 0.7716 | 9.39 |

These numbers are preliminary because several control runs are missing or partial.

### 5.2 Observations

On `5m_vs_6m`, the four-layer full AttnRes variant underperforms QMIX in final win rate and requires substantially more wall-clock time. The block variant is also worse. The two-layer variant is the most promising transfer setting: it has higher best win rate and higher AUC than QMIX, but its final win rate is still slightly lower.

On `3s5z`, most methods approach high performance, making the map less diagnostic. The lightweight variant again looks competitive, but this evidence is weaker because the baseline is close to saturation.

The current results support a cautious interpretation: heavy LLM-style depth-wise residual selection is not a reliable plug-in improvement for shallow recurrent MARL agents.

## 6. Discussion

The negative and mixed results are informative. The original AttnRes method targets residual stream dilution in deep LLMs. In contrast, the PyMARL agent is shallow and recurrent. Its main bottleneck is not depth-wise residual accumulation but coordination under partial observability.

This mismatch explains why the heavy variants are unstable. They introduce extra parameters, extra nonlinear transformations, and softmax depth selection into an already unstable value-learning setting. The two-layer version reduces this burden and therefore behaves better, but still does not consistently improve final win rate.

These findings suggest that transferring architectural ideas from LLMs to MARL requires adapting the role of attention. Instead of using attention to select among artificial depth sources after the GRU, a more natural MARL adaptation is to use attention for inter-agent communication.

## 7. Follow-Up Direction

A second-stage method should preserve the selection idea but move it to the agent dimension:

```text
agent GRU hidden states
-> agent-wise attention communication
-> residual or gated message fusion
-> individual action-value heads
-> QMIX / VDN / IQL
```

This aligns better with SMAC, where agents must coordinate based on local observations. The current AttnRes transfer study can motivate this redesign.

## 8. Conclusion

This paper studies direct transfer of Attention Residuals from LLMs to SMAC MARL agents. The results show that heavy direct transfer is not consistently beneficial and increases training cost. A lightweight two-layer variant is more stable and sometimes improves learning AUC, but the improvement is not strong enough to claim a robust algorithmic gain. These results indicate that LLM residual mechanisms should not be inserted into shallow recurrent MARL agents without task-specific adaptation. The next promising direction is to reinterpret attention residual selection as inter-agent communication rather than depth-wise residual replacement.

## TODO

- Complete missing `qmix_depth_mlp` runs.
- Complete cross-algorithm IQL/VDN lightweight transfer runs.
- Regenerate summary tables with `scripts/summarize_marl_transfer_adaptation.py`.
- Add learning curves for `5m_vs_6m`.
- Decide whether to include the follow-up communication method as a proposed extension or an additional positive experiment.
