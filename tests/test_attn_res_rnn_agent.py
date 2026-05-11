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
from modules.agents.attn_comm import AgentAttentionCommunication
from modules.agents.attncomm_rnn_agent import AttnCommRNNAgent
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
        "record_attn_weights": False,
        "attn_comm_enabled": True,
        "attn_comm_layers": 2,
        "attn_comm_hidden_dim": 64,
        "attn_comm_dropout": 0.0,
        "attn_comm_source_mode": "self_inclusive",
        "record_comm_attn_weights": False,
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

    def test_attention_weight_recording_is_optional_and_resettable(self):
        args = make_args(attn_res_layers=2, record_attn_weights=True)
        module = DepthwiseAttentionResidual(args.rnn_hidden_dim, args)
        hidden = th.randn(8, args.rnn_hidden_dim)

        module(hidden)
        stats = module.get_and_reset_attention_weight_stats()

        self.assertEqual(set(stats), {"attn_res_l0_src0", "attn_res_l1_src0", "attn_res_l1_src1"})
        self.assertAlmostEqual(stats["attn_res_l0_src0"], 1.0, places=6)
        self.assertAlmostEqual(stats["attn_res_l1_src0"] + stats["attn_res_l1_src1"], 1.0, places=6)
        self.assertEqual(module.get_and_reset_attention_weight_stats(), {})


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


class AgentAttentionCommunicationTest(unittest.TestCase):
    def test_self_inclusive_shapes_gradients_and_weight_sums(self):
        th.manual_seed(0)
        args = make_args(record_comm_attn_weights=True, attn_comm_source_mode="self_inclusive")
        module = AgentAttentionCommunication(args.rnn_hidden_dim, args)
        hidden = th.randn(2, args.n_agents, args.rnn_hidden_dim)

        out = module(hidden)
        self.assertEqual(out.shape, hidden.shape)
        self.assertFalse(th.isnan(out).any())

        out.sum().backward()
        grad_params = [
            module.q_proj[0].weight,
            module.k_proj[0].weight,
            module.v_proj[0].weight,
            module.out_proj[0].weight,
            module.gates[0].weight,
        ]
        for param in grad_params:
            self.assertIsNotNone(param.grad)
            self.assertTrue(th.isfinite(param.grad).all())

        stats = module.get_and_reset_attention_weight_stats()
        for layer in range(args.attn_comm_layers):
            for target in range(args.n_agents):
                row_sum = sum(stats["attn_comm_l{}_to{}_from{}".format(layer, target, source)] for source in range(args.n_agents))
                self.assertAlmostEqual(row_sum, 1.0, places=6)
        self.assertEqual(module.get_and_reset_attention_weight_stats(), {})

    def test_other_only_masks_self_attention(self):
        th.manual_seed(0)
        args = make_args(record_comm_attn_weights=True, attn_comm_source_mode="other_only")
        module = AgentAttentionCommunication(args.rnn_hidden_dim, args)
        hidden = th.randn(2, args.n_agents, args.rnn_hidden_dim)

        out = module(hidden)
        self.assertEqual(out.shape, hidden.shape)
        self.assertFalse(th.isnan(out).any())

        stats = module.get_and_reset_attention_weight_stats()
        for layer in range(args.attn_comm_layers):
            for target in range(args.n_agents):
                self_key = "attn_comm_l{}_to{}_from{}".format(layer, target, target)
                self.assertAlmostEqual(stats[self_key], 0.0, places=6)
                row_sum = sum(stats["attn_comm_l{}_to{}_from{}".format(layer, target, source)] for source in range(args.n_agents))
                self.assertAlmostEqual(row_sum, 1.0, places=6)


class AttnCommRNNAgentTest(unittest.TestCase):
    def test_forward_shapes_hidden_reuse_and_gradients(self):
        th.manual_seed(0)
        args = make_args(agent="attncomm_rnn", record_comm_attn_weights=True, attn_comm_source_mode="other_only")
        input_shape = 11
        batch_size = 2
        batch_agents = batch_size * args.n_agents
        agent = AttnCommRNNAgent(input_shape, args)

        hidden = agent.init_hidden().expand(batch_agents, -1)
        inputs = th.randn(batch_agents, input_shape)
        q, next_hidden = agent(inputs, hidden)

        self.assertEqual(q.shape, (batch_agents, args.n_actions))
        self.assertEqual(next_hidden.shape, (batch_agents, args.rnn_hidden_dim))
        self.assertFalse(th.isnan(q).any())
        self.assertFalse(th.isnan(next_hidden).any())

        inputs2 = th.randn(batch_agents, input_shape)
        q2, next_hidden2 = agent(inputs2, next_hidden)
        self.assertEqual(q2.shape, (batch_agents, args.n_actions))
        self.assertEqual(next_hidden2.shape, (batch_agents, args.rnn_hidden_dim))

        q2.sum().backward()
        grad_params = [
            agent.fc1.weight,
            agent.rnn.weight_ih,
            agent.attn_comm.q_proj[0].weight,
            agent.attn_comm.gates[0].weight,
            agent.fc2.weight,
        ]
        for param in grad_params:
            self.assertIsNotNone(param.grad)
            self.assertTrue(th.isfinite(param.grad).all())

        stats = agent.get_and_reset_attention_weight_stats()
        self.assertIn("attn_comm_l0_to0_from1", stats)
        self.assertEqual(agent.get_and_reset_attention_weight_stats(), {})


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

    def test_basic_mac_attncomm_agent_forward_shape(self):
        th.manual_seed(0)
        args = make_args(agent="attncomm_rnn", attn_comm_source_mode="self_inclusive", record_comm_attn_weights=True)
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
        self.assertIn("attn_comm_l0_to0_from0", mac.get_and_reset_attention_weight_stats())


if __name__ == "__main__":
    unittest.main()
