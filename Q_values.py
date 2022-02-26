import numpy as np
from NeuralNet import NeuralNet

# FILL THE CODE
# Enter inside the Q_values function and fill it with your code.
# You need to compute the Q values as output of your neural
# network. You can change the input of the function by adding other
# data, but the input of the function is suggested.
class Q_values:

    nn={}

    def __init__(self, layers):
        self.nn = NeuralNet(layers)

    # output, neuron_value
    def q_values(self, input):
        return self.nn.feedforward(input)

    def update_q_func(self, eta, neuron_value, a_agent, R, Q):
        delta = -(R - Q[a_agent])
        delta_v = np.zeros(self.nn.layers[len(self.nn.layers)-1])
        delta_v[a_agent] = delta
        self.nn.update(eta, neuron_value, delta_v)

    @staticmethod
    def encode_features(dfK2, s, check):
        s_k1check_onehot = np.array(s == 1).astype(float).reshape(-1)   # FEATURES FOR KING POSITION
        s_q1check_onehot = np.array(s == 2).astype(float).reshape(-1)   # FEATURES FOR QUEEN POSITION
        s_k2check_onehot = np.array(s == 3).astype(float).reshape(-1)   # FEATURE FOR ENEMY'S KING POSITION

        check_onehot=np.zeros([2])    # CHECK? FEATURE
        check_onehot[check]=1

        K2dof=np.zeros([8])   # NUMBER OF ALLOWED ACTIONS FOR ENEMY'S KING, ONE-HOT ENCODED
        K2dof[np.sum(dfK2).astype(int)]=1

        # ALL FEATURES...
        x = np.concatenate([s_k1check_onehot, s_q1check_onehot, s_k2check_onehot, check, K2dof],0)

        return x

