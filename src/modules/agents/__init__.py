REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .attnres_rnn_agent import AttnResRNNAgent
REGISTRY["attnres_rnn"] = AttnResRNNAgent


REGISTRY["attn_res_rnn"] = AttnResRNNAgent

from .depth_mlp_rnn_agent import DepthMLPRNNAgent
REGISTRY["depth_mlp_rnn"] = DepthMLPRNNAgent
