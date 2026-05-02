import torch.nn as nn
import torch.nn.functional as F


class DepthMLPRNNAgent(nn.Module):
    """RNN agent with an extra plain MLP stack after the GRU.

    This is the depth-only control for Attention Residuals experiments: it
    matches the extra post-GRU transform depth without adding cross-depth
    attention or learned residual source selection.
    """

    def __init__(self, input_shape, args):
        super(DepthMLPRNNAgent, self).__init__()
        self.args = args
        self.rnn_hidden_dim = args.rnn_hidden_dim

        depth_layers = int(getattr(args, "depth_mlp_layers", getattr(args, "attn_res_layers", 4)))
        depth_hidden_dim = int(getattr(args, "depth_mlp_hidden_dim", self.rnn_hidden_dim))
        dropout_p = float(getattr(args, "depth_mlp_dropout", 0.0))

        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)

        self.input_proj = nn.Identity() if depth_hidden_dim == self.rnn_hidden_dim else nn.Linear(
            self.rnn_hidden_dim, depth_hidden_dim
        )
        self.output_proj = nn.Identity() if depth_hidden_dim == self.rnn_hidden_dim else nn.Linear(
            depth_hidden_dim, self.rnn_hidden_dim
        )
        self.depth_stack = nn.ModuleList([
            nn.Sequential(
                nn.Linear(depth_hidden_dim, depth_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(depth_hidden_dim, depth_hidden_dim),
            )
            for _ in range(depth_layers)
        ])

        self.fc2 = nn.Linear(self.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        z = self.input_proj(h)
        for layer in self.depth_stack:
            z = F.relu(z + layer(z))

        q = self.fc2(self.output_proj(z))
        return q, h
