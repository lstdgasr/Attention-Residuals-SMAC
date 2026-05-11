import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class AgentAttentionCommunication(nn.Module):
    """Stacked agent-wise attention communication with gated residual fusion."""

    def __init__(self, input_dim, args):
        super(AgentAttentionCommunication, self).__init__()
        self.input_dim = input_dim
        self.n_agents = int(args.n_agents)
        self.enabled = bool(getattr(args, "attn_comm_enabled", False))
        self.layers = int(getattr(args, "attn_comm_layers", 0))
        self.hidden_dim = int(getattr(args, "attn_comm_hidden_dim", input_dim))
        self.dropout_p = float(getattr(args, "attn_comm_dropout", 0.0))
        self.source_mode = getattr(args, "attn_comm_source_mode", "self_inclusive")
        self.record_attn_weights = bool(getattr(args, "record_comm_attn_weights", False))

        if self.source_mode not in ("self_inclusive", "other_only"):
            raise ValueError("Unknown attn_comm_source_mode: {}".format(self.source_mode))

        self.active = self.enabled and self.layers > 0
        self._attn_weight_sums = {}
        self._attn_weight_counts = {}

        if self.active:
            self.norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(self.layers)])
            self.q_proj = nn.ModuleList([nn.Linear(input_dim, self.hidden_dim) for _ in range(self.layers)])
            self.k_proj = nn.ModuleList([nn.Linear(input_dim, self.hidden_dim) for _ in range(self.layers)])
            self.v_proj = nn.ModuleList([nn.Linear(input_dim, self.hidden_dim) for _ in range(self.layers)])
            self.out_proj = nn.ModuleList([nn.Linear(self.hidden_dim, input_dim) for _ in range(self.layers)])
            self.gates = nn.ModuleList([nn.Linear(input_dim * 2, input_dim) for _ in range(self.layers)])
            self.dropout = nn.Dropout(self.dropout_p)
            self._init_gate_bias()
        else:
            self.norms = nn.ModuleList()
            self.q_proj = nn.ModuleList()
            self.k_proj = nn.ModuleList()
            self.v_proj = nn.ModuleList()
            self.out_proj = nn.ModuleList()
            self.gates = nn.ModuleList()
            self.dropout = nn.Identity()

    def _init_gate_bias(self):
        for gate in self.gates:
            nn.init.constant_(gate.bias, -2.0)

    def forward(self, hidden):       #### 通信注意力模块             
        
        if not self.active:
            return hidden

        if hidden.dim() != 3:
            raise ValueError("AgentAttentionCommunication expects [batch, n_agents, hidden], got {}".format(tuple(hidden.shape)))
        if hidden.size(1) != self.n_agents:
            raise ValueError("Expected {} agents, got {}".format(self.n_agents, hidden.size(1)))

        out = hidden
        for layer_idx in range(self.layers):
            normed = self.norms[layer_idx](out)
            query = self.q_proj[layer_idx](normed)
            key = self.k_proj[layer_idx](normed)
            value = self.v_proj[layer_idx](out)

            logits = th.matmul(query, key.transpose(1, 2)) / math.sqrt(self.hidden_dim)
            logits = self._mask_logits(logits)
            weights = F.softmax(logits, dim=-1)
            weights = self._clean_masked_weights(weights)
            self._record_attention_weights(layer_idx, weights)

            message = th.matmul(weights, value)
            message = self.dropout(self.out_proj[layer_idx](message))
            gate = th.sigmoid(self.gates[layer_idx](th.cat([out, message], dim=-1)))
            out = out + gate * message

        return out

    def _mask_logits(self, logits):
        if self.source_mode != "other_only" or self.n_agents <= 1:
            return logits

        mask = th.eye(self.n_agents, dtype=th.bool, device=logits.device).unsqueeze(0)
        return logits.masked_fill(mask, -1e9)

    def _clean_masked_weights(self, weights):
        if self.source_mode != "other_only" or self.n_agents <= 1:
            return weights

        mask = th.eye(self.n_agents, dtype=th.bool, device=weights.device).unsqueeze(0)
        return weights.masked_fill(mask, 0.0)

    def _record_attention_weights(self, layer_idx, weights):
        if not self.record_attn_weights:
            return

        source_means = weights.detach().mean(dim=0).cpu()
        for target_idx in range(source_means.size(0)):
            for source_idx in range(source_means.size(1)):
                key = "attn_comm_l{}_to{}_from{}".format(layer_idx, target_idx, source_idx)
                value = float(source_means[target_idx, source_idx].item())
                self._attn_weight_sums[key] = self._attn_weight_sums.get(key, 0.0) + value
                self._attn_weight_counts[key] = self._attn_weight_counts.get(key, 0) + 1

    def get_and_reset_attention_weight_stats(self):
        if not self._attn_weight_sums:
            return {}

        stats = {
            key: self._attn_weight_sums[key] / self._attn_weight_counts[key]
            for key in sorted(self._attn_weight_sums)
            if self._attn_weight_counts.get(key, 0) > 0
        }
        self._attn_weight_sums = {}
        self._attn_weight_counts = {}
        return stats
