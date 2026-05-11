import torch.nn as nn
import torch.nn.functional as F

from .attn_comm import AgentAttentionCommunication


class AttnCommRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(AttnCommRNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.attn_comm = AgentAttentionCommunication(self.rnn_hidden_dim, args)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        batch_agents = inputs.size(0)
        if batch_agents % self.n_agents != 0:
            raise ValueError("Expected flattened batch to be divisible by n_agents={}, got {}".format(self.n_agents, batch_agents))

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)                                                                                       # 在每个 agent 输出 Q 值之前，让它先通过 attention 读取其他 agent 的 hidden states。

        batch_size = batch_agents // self.n_agents
        h_agents = h.view(batch_size, self.n_agents, self.rnn_hidden_dim)
        communicated = self.attn_comm(h_agents).reshape(batch_agents, self.rnn_hidden_dim)
        q = self.fc2(communicated)
        return q, h

    def get_and_reset_attention_weight_stats(self):
        return self.attn_comm.get_and_reset_attention_weight_stats()
