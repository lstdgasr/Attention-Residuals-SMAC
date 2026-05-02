# Follow-Up Design: Agent Attention Communication

## Motivation

The current direct transfer uses:

```text
GRU hidden -> depth-wise AttnRes -> Q head
```

This is not well aligned with SMAC. Attention Residuals address depth-wise residual dilution in deep LLMs, while SMAC mainly requires coordination across agents under partial observability.

The follow-up method should preserve the idea of selective aggregation but move it from the depth dimension to the agent dimension.

## Proposed Module

Input at each timestep:

```text
H = [h_1, h_2, ..., h_n]  # GRU hidden states for all agents
```

Communication:

```text
Q_i = W_q h_i
K_j = W_k h_j
V_j = W_v h_j
a_ij = softmax_j(Q_i K_j / sqrt(d))
m_i = sum_j a_ij V_j
```

Fusion:

```text
g_i = sigmoid(W_g [h_i, m_i])
h'_i = h_i + g_i * m_i
q_i = QHead(h'_i)
```

Then keep the existing MARL backend:

```text
individual q_i -> QMIX / VDN / IQL
```

## Suggested Names

```text
AttnComm-QMIX
Residual Communication QMIX
Agent Attention Residual Network
```

## Minimal Experiment

If the adaptation-study paper needs a positive extension, run only:

```text
5m_vs_6m: qmix vs qmix_attnres_l2 vs attncomm_qmix
3s5z: qmix vs qmix_attnres_l2 vs attncomm_qmix
seeds: 1, 2, 3
```

## Paper Framing

Use the current negative/mixed results as motivation:

```text
Direct depth-wise transfer is unstable because shallow recurrent MARL agents do not suffer from the same residual-stream problem as deep LLMs. A better transfer is to reinterpret selective attention as agent-wise communication, which directly targets coordination.
```

## Implementation Notes

- Add the module in the controller or in a new multi-agent-aware agent wrapper, because communication needs all agent hidden states at the same timestep.
- Do not implement it inside the current per-agent `RNNAgent.forward`, because that function only sees flattened per-agent inputs independently.
- Keep the mixer unchanged for the first version.
- Use a residual gate to avoid destabilizing early training.
- Start with one attention head and hidden dimension 64.
