import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(th.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class DepthwiseAttentionResidual(nn.Module):
    """Attention Residuals adapted to the hidden state of a PyMARL RNN agent.

    The module treats a small stack of MLP transforms as depth sources and
    replaces fixed residual accumulation with softmax attention over previous
    sources. Learned pseudo-queries are zero-initialised, matching the paper's
    stable-start recommendation.
    """

    def __init__(self, input_dim, args):
        super(DepthwiseAttentionResidual, self).__init__()
        self.input_dim = input_dim
        self.enabled = getattr(args, "attn_res_enabled", False)
        self.mode = getattr(args, "attn_res_mode", getattr(args, "attn_res_impl", "identity"))
        if self.mode == "depth_attention":
            self.mode = "full"

        self.layers = int(getattr(args, "attn_res_layers", 0))
        self.hidden_dim = int(getattr(args, "attn_res_hidden_dim", input_dim))
        self.block_size = int(getattr(args, "attn_res_block_size", 0))
        self.dropout_p = float(getattr(args, "attn_res_dropout", 0.0))
        self.rms_eps = float(getattr(args, "attn_res_rms_eps", 1e-8))
        self.record_attn_weights = bool(getattr(
            args,
            "record_attn_weights",
            getattr(args, "attn_res_record_weights", False),
        ))
        self._attn_weight_sums = {}
        self._attn_weight_counts = {}

        self.active = self.enabled and self.mode in ("full", "block") and self.layers > 0

        self.input_proj = nn.Identity() if self.hidden_dim == input_dim else nn.Linear(input_dim, self.hidden_dim)
        self.output_proj = nn.Identity() if self.hidden_dim == input_dim else nn.Linear(self.hidden_dim, input_dim)

        if self.active:
            self.queries = nn.Parameter(th.zeros(self.layers, self.hidden_dim))
            self.norm = RMSNorm(self.hidden_dim, eps=self.rms_eps)
            self.transforms = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in range(self.layers)
            ])
            self.dropout = nn.Dropout(self.dropout_p)
        else:
            self.register_parameter("queries", None)
            self.norm = None
            self.transforms = nn.ModuleList()
            self.dropout = nn.Identity()

    def forward(self, hidden):
        if not self.active:
            return hidden

        hidden = self.input_proj(hidden)
        if self.mode == "block":
            out = self._forward_block(hidden)
        else:
            out = self._forward_full(hidden)
        return self.output_proj(out)

    def _attend(self, layer_idx, values):
        value_stack = th.stack(values, dim=0)
        key_stack = self.norm(value_stack)
        logits = th.einsum("nbd,d->nb", key_stack, self.queries[layer_idx])
        logits = logits / math.sqrt(self.hidden_dim)
        weights = F.softmax(logits, dim=0).unsqueeze(-1)
        self._record_attention_weights(layer_idx, weights)
        return (weights * value_stack).sum(dim=0)

    def _record_attention_weights(self, layer_idx, weights):
        if not self.record_attn_weights:
            return

        source_means = weights.detach().squeeze(-1).mean(dim=1).cpu()
        for source_idx, value in enumerate(source_means):
            key = "attn_res_l{}_src{}".format(layer_idx, source_idx)
            self._attn_weight_sums[key] = self._attn_weight_sums.get(key, 0.0) + float(value.item())
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

    def _forward_full(self, hidden):
        sources = [hidden]
        out = hidden
        for layer_idx in range(self.layers):
            attended = self._attend(layer_idx, sources)
            layer_out = self.dropout(self.transforms[layer_idx](attended))
            sources.append(layer_out)
            out = attended + layer_out
        return out

    def _forward_block(self, hidden):
        block_size = self.block_size if self.block_size > 0 else self.layers
        completed_blocks = [hidden]
        partial_block = None
        out = hidden

        for layer_idx in range(self.layers):
            values = list(completed_blocks)
            if partial_block is not None:
                values.append(partial_block)

            attended = self._attend(layer_idx, values)
            layer_out = self.dropout(self.transforms[layer_idx](attended))
            partial_block = layer_out if partial_block is None else partial_block + layer_out
            out = attended + layer_out

            is_boundary = ((layer_idx + 1) % block_size == 0) or (layer_idx + 1 == self.layers)
            if is_boundary:
                completed_blocks.append(partial_block)
                partial_block = None

        return out
