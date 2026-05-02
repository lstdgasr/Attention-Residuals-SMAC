import os
import sys
import unittest
from types import SimpleNamespace as SN

import torch as th


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from components.episode_buffer import EpisodeBatch
from controllers.basic_controller import BasicMAC
from modules.agents.attn_residual import DepthwiseAttentionResidual
from modules.agents.attnres_rnn_agent import AttnResRNNAgent
from modules.agents.depth_mlp_rnn_agent import DepthMLPRNNAgent


def make_args(**overrides):
    args = {
        "rnn_hidden_dim": 64,
        "attn_res_enabled": True,
        "attn_res_mode": "full",
        "attn_res_layers": 4,
        "attn_res_hidden_dim": 64,
        "attn_res_block_size": 0,
        "attn_res_dropout": 0.0,
        "attn_res_rms_eps": 1e-8,
        "depth_mlp_layers": 4,
        "depth_mlp_hidden_dim": 64,
        "depth_mlp_dropout": 0.0,
        "n_actions": 5,
        "n_agents": 3,
        "agent": "attnres_rnn",
        "agent_output_type": "q",
        "action_selector": "epsilon_greedy",
        "epsilon_start": 1.0,
        "epsilon_finish": 0.05,
        "epsilon_anneal_time": 50000,
        "obs_agent_id": True,
        "obs_last_action": True,
        "test_greedy": True,
    }
    args.update(overrides)
    return SN(**args)


class DepthwiseAttentionResidualTest(unittest.TestCase):
    def assert_mode_shapes_and_gradients(self, mode, block_size=0):
        th.manual_seed(0)
        args = make_args(attn_res_mode=mode, attn_res_block_size=block_size)
        module = DepthwiseAttentionResidual(args.rnn_hidden_dim, args)
        hidden = th.randn(8, args.rnn_hidden_dim)

        out = module(hidden)
        self.assertEqual(out.shape, hidden.shape)
        self.assertFalse(th.isnan(out).any())

        self.assertIsNotNone(module.queries)
        self.assertTrue(th.equal(module.queries.data, th.zeros_like(module.queries.data)))

        out.sum().backward()
        self.assertIsNotNone(module.queries.grad)
        self.assertTrue(th.isfinite(module.queries.grad).all())
        self.assertIsNotNone(module.transforms[0][0].weight.grad)
        self.assertTrue(th.isfinite(module.transforms[0][0].weight.grad).all())
        self.assertIsNotNone(module.norm.weight.grad)
        self.assertTrue(th.isfinite(module.norm.weight.grad).all())

    def test_full_mode_shapes_gradients_and_zero_queries(self):
        self.assert_mode_shapes_and_gradients("full")

    def test_block_mode_shapes_gradients_and_zero_queries(self):
        self.assert_mode_shapes_and_gradients("block", block_size=2)

    def test_identity_mode_is_noop(self):
        args = make_args(attn_res_enabled=False, attn_res_mode="identity")
        module = DepthwiseAttentionResidual(args.rnn_hidden_dim, args)
        hidden = th.randn(8, args.rnn_hidden_dim)
        out = module(hidden)
        self.assertTrue(th.equal(out, hidden))


class AttnResRNNAgentTest(unittest.TestCase):
    def assert_forward_shapes_and_gradients(self, attn_res_layers):
        th.manual_seed(0)
        args = make_args(attn_res_layers=attn_res_layers)
        input_shape = 11
        batch_agents = 8
        agent = AttnResRNNAgent(input_shape, args)

        hidden = agent.init_hidden()
        self.assertEqual(hidden.shape, (1, args.rnn_hidden_dim))

        inputs = th.randn(batch_agents, input_shape)
        hidden = hidden.expand(batch_agents, -1)
        q, next_hidden = agent(inputs, hidden)

        self.assertEqual(q.shape, (batch_agents, args.n_actions))
        self.assertEqual(next_hidden.shape, (batch_agents, args.rnn_hidden_dim))
        self.assertFalse(th.isnan(q).any())
        self.assertFalse(th.isnan(next_hidden).any())

        q.sum().backward()
        grad_params = [
            agent.fc1.weight,
            agent.rnn.weight_ih,
            agent.attn_residual.queries,
            agent.attn_residual.transforms[0][0].weight,
            agent.fc2.weight,
        ]
        for param in grad_params:
            self.assertIsNotNone(param.grad)
            self.assertTrue(th.isfinite(param.grad).all())

    def test_forward_shapes_and_gradients_one_layer(self):
        self.assert_forward_shapes_and_gradients(attn_res_layers=1)

    def test_forward_shapes_and_gradients_four_layers(self):
        self.assert_forward_shapes_and_gradients(attn_res_layers=4)


class DepthMLPRNNAgentTest(unittest.TestCase):
    def test_forward_shapes_and_gradients(self):
        th.manual_seed(0)
        args = make_args(agent="depth_mlp_rnn", depth_mlp_layers=4)
        input_shape = 11
        batch_agents = 8
        agent = DepthMLPRNNAgent(input_shape, args)

        hidden = agent.init_hidden().expand(batch_agents, -1)
        inputs = th.randn(batch_agents, input_shape)
        q, next_hidden = agent(inputs, hidden)

        self.assertEqual(q.shape, (batch_agents, args.n_actions))
        self.assertEqual(next_hidden.shape, (batch_agents, args.rnn_hidden_dim))
        self.assertFalse(th.isnan(q).any())
        self.assertFalse(th.isnan(next_hidden).any())

        q.sum().backward()
        grad_params = [
            agent.fc1.weight,
            agent.rnn.weight_ih,
            agent.depth_stack[0][0].weight,
            agent.fc2.weight,
        ]
        for param in grad_params:
            self.assertIsNotNone(param.grad)
            self.assertTrue(th.isfinite(param.grad).all())


class AttnResBasicMACTest(unittest.TestCase):
    def test_basic_mac_forward_shape_and_hidden_state(self):
        th.manual_seed(0)
        args = make_args()
        batch_size = 2
        max_seq_length = 3
        obs_shape = 4
        groups = {"agents": args.n_agents}
        scheme = {
            "obs": {"vshape": obs_shape, "group": "agents"},
            "actions_onehot": {"vshape": (args.n_actions,), "group": "agents"},
            "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": th.int},
        }
        batch = EpisodeBatch(scheme, groups, batch_size, max_seq_length)
        batch.update({
            "obs": th.randn(batch_size, args.n_agents, obs_shape).tolist(),
            "avail_actions": th.ones(batch_size, args.n_agents, args.n_actions).int().tolist(),
        }, ts=0)

        mac = BasicMAC(batch.scheme, groups, args)
        mac.init_hidden(batch_size)
        out = mac.forward(batch, t=0)

        self.assertEqual(out.shape, (batch_size, args.n_agents, args.n_actions))
        self.assertEqual(mac.hidden_states.shape, (batch_size * args.n_agents, args.rnn_hidden_dim))
        self.assertFalse(th.isnan(out).any())
        self.assertFalse(th.isnan(mac.hidden_states).any())

    def test_basic_mac_depth_mlp_agent_forward_shape(self):
        th.manual_seed(0)
        args = make_args(agent="depth_mlp_rnn")
        batch_size = 2
        max_seq_length = 3
        obs_shape = 4
        groups = {"agents": args.n_agents}
        scheme = {
            "obs": {"vshape": obs_shape, "group": "agents"},
            "actions_onehot": {"vshape": (args.n_actions,), "group": "agents"},
            "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": th.int},
        }
        batch = EpisodeBatch(scheme, groups, batch_size, max_seq_length)
        batch.update({
            "obs": th.randn(batch_size, args.n_agents, obs_shape).tolist(),
            "avail_actions": th.ones(batch_size, args.n_agents, args.n_actions).int().tolist(),
        }, ts=0)

        mac = BasicMAC(batch.scheme, groups, args)
        mac.init_hidden(batch_size)
        out = mac.forward(batch, t=0)

        self.assertEqual(out.shape, (batch_size, args.n_agents, args.n_actions))
        self.assertFalse(th.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
