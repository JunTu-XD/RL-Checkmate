import numpy as np


class NeuralNet:
    W_bias = []
    current_neuron_value = []
    layers = []

    momentum = 0.8
    cached_momentum_change = []
    derivative = []
    def __init__(self, layers):
        self.layers = layers
        self.cached_momentum_change = []
        self.W_bias = self.init_W_bias()
        self.momentum = 0.8
        return

    ## h = W.T ( N_h, N_pre) * Pre_in(N_pre * 1)
    def init_W_bias(self):
        W_b = []
        for layer_idx in range(len(self.layers)-1):
            W = np.random.uniform(-1, 1, [self.layers[layer_idx], self.layers[layer_idx+1]]) / (self.layers[layer_idx] + self.layers[layer_idx+1])
            b = np.zeros([ self.layers[layer_idx+1]])
            W_b.append((W,b))
            self.cached_momentum_change.append((np.zeros(W.shape),np.zeros(b.shape)))
        return W_b

    ## ReLu
    def activation_func(self, h):
        return np.max([np.zeros(h.shape[0]), h], axis=0)

    @staticmethod
    def derivative_ReLu(h):
        d_a_d_h = np.zeros((h.shape[0], h.shape[0]))
        diagonal = np.copy(h)
        diagonal[np.where(diagonal > 0)] = 1
        diagonal[np.where(diagonal <= 0)] = 0
        np.fill_diagonal(d_a_d_h, diagonal)
        return d_a_d_h

    # target: N_a * 1
    # output: N_a * 1
    # return: [(d_w, d_b, d_a)]
    # neuron_values: [(input, input), (h,a),(h,a), output]
    def partial_derivative(self, W_bias, delta, neuron_values):
        res = []
        for i in range(len(W_bias)):
            res.append((0, 0))

        prev_idx = len(neuron_values) - 3
        curr_idx = len(neuron_values) - 2
        d_q_loss_d_a = delta

        d_y_d_a_prev = d_q_loss_d_a
        d_y_d_a_prev = np.reshape(d_y_d_a_prev, (d_y_d_a_prev.shape[0], 1)).transpose()
        for i in range(0, len(W_bias)):
            w_idx = len(W_bias) - i - 1
            (w, b) = W_bias[w_idx]
            (h, a) = neuron_values[curr_idx]
            (prev_h, prev_a) = neuron_values[prev_idx]

            # d_a / d_h
            d_a_d_h = self.derivative_ReLu(h)

            # dh / dw & db
            d_h_d_w = np.reshape(prev_a, (prev_a.shape[0], 1))
            # h*b h=b
            d_h_d_b = np.zeros((h.shape[0], b.shape[0]))
            np.fill_diagonal(d_h_d_b, np.ones(b.shape[0]))

            d_y_d_h = d_y_d_a_prev.dot(d_a_d_h)
            d_y_d_b = d_y_d_h.dot(d_h_d_b).transpose()
            res[w_idx] = (d_h_d_w.dot(d_y_d_h), d_y_d_b.reshape((d_y_d_b.shape[0],)), d_y_d_a_prev)

            if prev_idx > 0:
                # d_h / d_prev_a = W[w_idx-1]
                d_h_d_prev_a = W_bias[w_idx][0].transpose()
                # d_y_d / d_a_prev = (d_y/d_a) .dot (d_a / d_h) .dot (d_h / d_prev a)
                d_y_d_a_prev = d_y_d_h.dot(d_h_d_prev_a)

            prev_idx -= 1
            curr_idx -= 1
        self.derivative=np.copy(res)
        return res

    #
    # [(input,input)# for unifying, (h, a), (h,a), output]
    #
    def feedforward(self, input):
        neuron_value = []
        prev = np.asarray(input)
        neuron_value.append((prev, prev))
        W_bias = self.W_bias

        for (w, bias) in W_bias:
            # vector.dot w => vector
            h = prev.dot(np.asarray(w))
            h = h + bias
            activation = self.activation_func(h)
            neuron_value.append((h, activation))
            prev = activation

        output = prev
        neuron_value.append(output)

        return output, neuron_value

    def update(self, eta, neuron_value, delta):
        derivative = self.partial_derivative(self.W_bias, delta, neuron_value)
        idx = 0
        for wb in self.W_bias:
            new_w = wb[0] - eta * derivative[idx][0]
            new_b = wb[1] - eta * derivative[idx][1]
            self.W_bias[idx] = (new_w, new_b)
            idx+=1
        return self.W_bias

    def momentum_gradient_decent(self, eta, delta, neuron_value):
        derivative = self.partial_derivative(self.W_bias, delta, neuron_value)
        idx = 0
        for wb in self.W_bias:
            w_change = eta * derivative[idx][0] + self.momentum * self.cached_momentum_change[idx][0]
            b_change = eta * derivative[idx][1] + self.momentum * self.cached_momentum_change[idx][1]
            new_w = wb[0] - w_change
            new_b = wb[1] - b_change
            self.W_bias[idx] = (new_w, new_b)
            self.cached_momentum_change[idx] = (w_change, b_change)
            idx+=1
        return self.W_bias