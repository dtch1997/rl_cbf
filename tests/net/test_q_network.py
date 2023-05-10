import torch
import unittest
import numpy as np
from unittest.mock import MagicMock
from torch.autograd import gradcheck
from rl_cbf.net.q_network import QNetwork, QNetworkEnsemble

class MockEnv:
    def __init__(self):
        self.single_observation_space = MagicMock()
        self.single_observation_space.shape = [4]
        self.single_action_space = MagicMock()
        self.single_action_space.n = 2

class TestQNetwork(unittest.TestCase):
    def test_forward(self):
        env = MockEnv()
        net = QNetwork(env, device='cpu')
        x = torch.randn(1, 4).to('cpu')
        self.assertEqual(net(x).shape, (1, 2))

    def test_predict(self):
        env = MockEnv()
        net = QNetwork(env, device='cpu')
        states = np.random.randn(1, 4)
        self.assertEqual(net.predict(states).shape, (1, 2))

    def test_predict_value(self):
        env = MockEnv()
        net = QNetwork(env, device='cpu')
        states = np.random.randn(1, 4)
        self.assertEqual(net.predict_value(states).shape, (1,))

    def test_predict_action(self):
        env = MockEnv()
        net = QNetwork(env, device='cpu')
        states = np.random.randn(1, 4)
        self.assertEqual(net.predict_action(states).shape, (1,))

class TestQNetworkEnsemble(unittest.TestCase):
    def test_forward(self):
        env = MockEnv()
        net1 = QNetwork(env)
        net2 = QNetwork(env)
        net_ensemble = QNetworkEnsemble(env, [net1, net2])
        x = torch.randn(1, 4)
        self.assertEqual(net_ensemble(x).shape, (1, 2))

    def test_get_num_models(self):
        env = MockEnv()
        net1 = QNetwork(env)
        net2 = QNetwork(env)
        net_ensemble = QNetworkEnsemble(env, [net1, net2])
        self.assertEqual(net_ensemble.get_num_models(), 2)

if __name__ == '__main__':
    unittest.main()
