import numpy as np
from NeuralNet import NeuralNet

# FILL THE CODE
# Enter inside the Q_values function and fill it with your code.
# You need to compute the Q values as output of your neural
# network. You can change the input of the function by adding other
# data, but the input of the function is suggested.
class Q_values:

    nn={}
    gamma=0.85
    def __init__(self, layers):
        self.nn = NeuralNet(layers)

    ### similar to comments on SARSA
    def q_values(self, input):
        return self.nn.feedforward(input)
    ### gradient decent update
    ### Q_s is the current function output of current state value
    def update_q_func(self, eta, neuron_value, a_agent, R, Q_s):
        delta = -(R - Q_s[a_agent])
        delta_v = np.zeros(self.nn.layers[len(self.nn.layers)-1])
        delta_v[a_agent] = delta
        self.nn.momentum_gradient_decent(eta, delta_v, neuron_value)

    ### calculate Q value of future
    def calculate_next_Q(self,Q_next, allowed_next, next_state):
        allow_v = np.copy(Q_next)
        allow_v[np.where(allowed_next.flatten()!=1)] = 0
        next_action_value = 0
        ### defensive programming in case that all value of allowed action is 0
        if np.max(allow_v) != 0:
            next_action_value = Q_next[np.argmax(allow_v)]
        return self.gamma * next_action_value
