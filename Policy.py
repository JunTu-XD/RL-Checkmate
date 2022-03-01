import numpy as np

def epsilon_greedy(v_values, allowed_action, epsilon = 0.2):

    rand_a = np.random.uniform(0, 1) < epsilon
    # random
    if rand_a == 1:
        a_agent = np.random.permutation(np.where(allowed_action.flatten()==1)[0])[0]
    # greedy
    else:
        allow_v = np.copy(v_values)
        allow_v[np.where(allowed_action.flatten()!=1)] = 0
        if (np.max(allow_v) == 0):
            return np.random.permutation(np.where(allowed_action.flatten()==1)[0])[0]
        else:
            return np.argmax(allow_v)
    return a_agent
