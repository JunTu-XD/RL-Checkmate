import numpy as np
from NeuralNet import NeuralNet
from Policy import *

class sarsa:
    nn={}
    gamma=0.85
    epsilon = 0.2
    # init neural net
    def __init__(self, layers):
        self.nn = NeuralNet(layers)
    # return function's output, feed value into nets
    def sarsa_value(self, input):
        return self.nn.feedforward(input)
    # calculate delta of R, do gradient decent
    # neuron_value is the value of each neuron, then used to calculate gradient.
    # a_agent: action of current state
    # R: immediate reward + gamma*future reward
    # SV_s: state value calculated by function
    def update_sarsa_func(self, eta, neuron_value, a_agent, R, SV_s):
        ## when put into gradient decent, 
        ## W - eta * derivatives equals W + eta * (-delta) * other partial_derivatives (chain rule)
        delta = -(R - SV_s[a_agent])
        ## only update parts associated with the action
        delta_v = np.zeros(self.nn.layers[len(self.nn.layers)-1])
        delta_v[a_agent] = delta
        self.nn.momentum_gradient_decent(eta, delta_v, neuron_value)
        
    # follow the epsilon_greedy to select the future action and return the value
    # multiply gamma
    # will be used outside this class to calculate r + gamma * future reward
    ### SV_next is the action value table of the next state
    ### allowed_next is the allowed action in the next state
    ### next_state is unused in this case. 
    ###### Since I will use a lambda expression outside to pass how each agent calculate future value, this is an placeholder for interface but not using explicit interface defination
    def calculate_next_V(self, SV_next, allowed_next, next_state):
        a_next = epsilon_greedy(SV_next, allowed_next, self.epsilon)
        return self.gamma * SV_next[a_next]