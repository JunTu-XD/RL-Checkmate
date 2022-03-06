from Q_values import Q_values
import numpy as np
import copy

class Double_Q:
    Q_primary ={}
    Q_target = {}
    update_coefficient = 0.05
    gamma = 0.85
    def __init__(self, layers):
        self.Q_primary = Q_values(layers)
        self.Q_target = Q_values(layers)
        self.Q_target.nn.W_bias = copy.deepcopy(self.Q_primary.nn.W_bias)

    def update_func(self, eta, neuron_value, a_agent, R, Q_s):
        delta = -(R - Q_s[a_agent])
        delta_v = np.zeros(self.Q_primary.nn.layers[len(self.Q_primary.nn.layers)-1])
        delta_v[a_agent] = delta
        W_bias_primary = self.Q_primary.nn.momentum_gradient_decent(eta, delta_v, neuron_value)
        ## calulate
        for idx in range(len(self.Q_target.nn.W_bias)):
            (w_p,b_p) = W_bias_primary[idx]
            (w_t, b_t) = self.Q_target.nn.W_bias[idx]
            nw = (1- self.update_coefficient) * w_t + self.update_coefficient * w_p
            nb = (1- self.update_coefficient) * b_t + self.update_coefficient * b_p
            self.Q_target.nn.W_bias[idx] = (nw, nb)

    def q_primary_value(self, input):
        return self.Q_target.q_values(input)
    def q_target_value(self, input):
        return self.Q_primary.q_values(input)

    def next_action_Q(self, Q_next, allowed_next,  next_state):
        allow_v = np.copy(Q_next)
        allow_v[np.where(allowed_next.flatten()!=1)] = 0
        if (np.max(allow_v) == 0):
            action = np.random.permutation(np.where(allowed_next.flatten()==1)[0])[0]
        else:
            action = np.argmax(allow_v)
        ## use primary online network to calculate next action
        ## use target online network to evaluation next action's value
        return self.gamma * self.Q_target.q_values(next_state)[0][action]



