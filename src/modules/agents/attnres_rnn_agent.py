import torch.nn as nn
import torch.nn.functional as F

from .attn_residual import DepthwiseAttentionResidual


class AttnResRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(AttnResRNNAgent, self).__init__()
        self.args = args

        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.attn_residual = DepthwiseAttentionResidual(self.rnn_hidden_dim, args)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(self.attn_residual(h))
        return q, h

    def get_and_reset_attention_weight_stats(self):
        return self.attn_residual.get_and_reset_attention_weight_stats()
