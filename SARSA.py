import numpy as np
from NeuralNet import NeuralNet
from Policy import *

class sarsa:
    nn={}
    gamma=0.85
    epsilon = 0.2

    def __init__(self, layers):
        self.nn = NeuralNet(layers)
    # output, neuron_value
    def sarsa_value(self, input):
        return self.nn.feedforward(input)

    def update_sarsa_func(self, eta, neuron_value, a_agent, R, SV_s):
        delta = -(R - SV_s[a_agent])
        delta_v = np.zeros(self.nn.layers[len(self.nn.layers)-1])
        delta_v[a_agent] = delta
        self.nn.momentum_gradient_decent(eta, delta_v, neuron_value)

    def calculate_next_V(self, SV_next, state_next):
        a_next = epsilon_greedy(SV_next, state_next, self.epsilon)
        return self.gamma * SV_next[a_next]