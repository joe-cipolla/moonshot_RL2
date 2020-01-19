import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from rl_glue import RLGlue
from environment import BaseEnvironment

from lunar_lander import LunarLanderEnvironment

from agent import BaseAgent

from collections import deque

from copy import deepcopy

from tqdm import tqdm

import os 

import shutil

from plot_script import plot_result


# Action-Value Network
# Neural network function approximator for Agent.

class ActionValueNetwork:
    def __init__(self, network_config):
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")
        
        self.rand_generator = np.random.RandomState(network_config.get("seed"))
        
        # Specify self.layer_size which shows the number of nodes in each layer
        self.layer_sizes = [ self.state_dim, self.num_hidden_units, self.num_actions ]
        
        # Initialize the weights of the neural network
        # self.weights is an array of dictionaries with each dictionary corresponding to 
        # the weights from one layer to the next. Each dictionary includes W and b
        self.weights = [dict() for i in range(0, len(self.layer_sizes) - 1)]
        for i in range(0, len(self.layer_sizes) - 1):
            self.weights[i]['W'] = self.init_saxe(self.layer_sizes[i], self.layer_sizes[i + 1])
            self.weights[i]['b'] = np.zeros((1, self.layer_sizes[i + 1]))
    
    def get_action_values(self, s):
        """
        Args:
            s (Numpy array): The state.
        Returns:
            The action-values (Numpy array) calculated using the network's weights.
        """
        
        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        psi = np.dot(s, W0) + b0
        x = np.maximum(psi, 0)
        
        W1, b1 = self.weights[1]['W'], self.weights[1]['b']
        q_vals = np.dot(x, W1) + b1

        return q_vals
    
    def get_TD_update(self, s, delta_mat):
        """
        Args:
            s (Numpy array): The state.
            delta_mat (Numpy array): A 2D array of shape (batch_size, num_actions). Each row of delta_mat  
            correspond to one state in the batch. Each row has only one non-zero element 
            which is the TD-error corresponding to the action taken.
        Returns:
            The TD update (Array of dictionaries with gradient times TD errors) for the network's weights
        """

        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        W1, b1 = self.weights[1]['W'], self.weights[1]['b']
        
        psi = np.dot(s, W0) + b0
        x = np.maximum(psi, 0)
        dx = (psi > 0).astype(float)

        # td_update has the same structure as self.weights, that is an array of dictionaries.
        # td_update[0]["W"], td_update[0]["b"], td_update[1]["W"], and td_update[1]["b"] have the same shape as 
        # self.weights[0]["W"], self.weights[0]["b"], self.weights[1]["W"], and self.weights[1]["b"] respectively
        td_update = [dict() for i in range(len(self.weights))]
         
        v = delta_mat
        td_update[1]['W'] = np.dot(x.T, v) * 1. / s.shape[0]
        td_update[1]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]
        
        v = np.dot(v, W1.T) * dx
        td_update[0]['W'] = np.dot(s.T, v) * 1. / s.shape[0]
        td_update[0]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]
                
        return td_update
    

    def init_saxe(self, rows, cols):
        """
        Args:
            rows (int): number of input units for layer.
            cols (int): number of output units for layer.
        Returns:
            NumPy Array consisting of weights for the layer based on the initialization in Saxe et al.
        """
        tensor = self.rand_generator.normal(0, 1, (rows, cols))
        if rows < cols:
            tensor = tensor.T
        tensor, r = np.linalg.qr(tensor)
        d = np.diag(r, 0)
        ph = np.sign(d)
        tensor *= ph

        if rows < cols:
            tensor = tensor.T
        return tensor
    
    def get_weights(self):
        """
        Returns: 
            A copy of the current weights of this network.
        """
        return deepcopy(self.weights)
    
    def set_weights(self, weights):
        """
        Args: 
            weights (list of dictionaries): Consists of weights that this network will set as its own weights.
        """
        self.weights = deepcopy(weights)



## Test Code for ActionValueNetwork __init__() ## 
#
# NOTE: The test below is limited in scope. Additional tests are used in the autograder, so it is recommended 
# to test your implementations more carefully for correctness.
network_config = {
    "state_dim": 5,
    "num_hidden_units": 20,
    "num_actions": 3
}

test_network = ActionValueNetwork(network_config)
print("layer_sizes:", test_network.layer_sizes)
assert(np.allclose(test_network.layer_sizes, np.array([5, 20, 3])))

print("Passed the asserts! (Note: These are however limited in scope, additional testing is encouraged.)")



# Adam Optimizer
#
# The Adam algorithm is a more advanced variant of stochastic gradient descent (SGD).
# The Adam algorithm improves the SGD update with two concepts: adaptive vector stepsizes and momentum. It keeps 
# running estimates of the mean and second moment of the updates, denoted by m and v respectively.

class Adam():
    def __init__(self, layer_sizes, 
                 optimizer_info):
        self.layer_sizes = layer_sizes

        # Specify Adam algorithm's hyper parameters
        self.step_size = optimizer_info.get("step_size")
        self.beta_m = optimizer_info.get("beta_m")  # mean
        self.beta_v = optimizer_info.get("beta_v")  # 2nd moment
        self.epsilon = optimizer_info.get("epsilon")
        
        # Initialize Adam algorithm's m and v
        self.m = [dict() for i in range(1, len(self.layer_sizes))]
        self.v = [dict() for i in range(1, len(self.layer_sizes))]
        
        for i in range(0, len(self.layer_sizes) - 1)
            self.m[i]["W"] = np.zeros((self.layer_sizes[i],self.layer_sizes[i+1]))
            self.m[i]["b"] = np.zeros((1, self.layer_sizes[i + 1]))
            self.v[i]["W"] = np.zeros((self.layer_sizes[i],self.layer_sizes[i+1]))
            self.v[i]["b"] = np.zeros((1, self.layer_sizes[i + 1]))
            
        # Notice that to calculate m_hat and v_hat, we use powers of beta_m and beta_v to 
        # the time step t. We can calculate these powers using an incremental product. At initialization then, 
        # beta_m_product and beta_v_product should be ...? (Note that timesteps start at 1 and if we were to 
        # start from 0, the denominator would be 0.)
        self.beta_m_product = self.beta_m
        self.beta_v_product = self.beta_v
    
    def update_weights(self, weights, td_errors_times_gradients):
        """
        Args:
            weights (Array of dictionaries): The weights of the neural network.
            td_errors_times_gradients (Array of dictionaries): The gradient of the 
            action-values with respect to the network's weights times the TD-error
        Returns:
            The updated weights (Array of dictionaries).
        """
        for i in range(len(weights)):
            for param in weights[i].keys():
                self.m[i][param] = (self.beta_m * self.m[i][param]) + ((1 - self.beta_m )*td_errors_times_gradients[i][param])
                self.v[i][param] = (self.beta_v * selfv[i][param]) + ((1 - self.beta_v )*(td_errors_times_gradients[i][param] ** 2))
                m_hat = self.m[i][param] / (1 - self.beta_m_product)
                v_hat = self.v[i][param] / (1 - self.beta_v_product)
                weight_update = (self.step_size / (np.sqrt(v_hat) + self.epsilon) ) * m_hat
                    
                weights[i][param] = weights[i][param] + weight_update
        # Notice that to calculate m_hat and v_hat, we use powers of beta_m and beta_v to 
        ### update self.beta_m_product and self.beta_v_product
        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v
        
        return weights


## Test Code for Adam __init__() ##
#
# NOTE: The test below is limited in scope. Additional tests are used in the autograder, so it is recommended 
# to test your implementations more carefully for correctness.
network_config = {"state_dim": 5,
                  "num_hidden_units": 2,
                  "num_actions": 3
                 }

optimizer_info = {"step_size": 0.1,
                  "beta_m": 0.99,
                  "beta_v": 0.999,
                  "epsilon": 0.0001
                 }

network = ActionValueNetwork(network_config)
test_adam = Adam(network.layer_sizes, optimizer_info)

print("m[0][\"W\"] shape: {}".format(test_adam.m[0]["W"].shape))
print("m[0][\"b\"] shape: {}".format(test_adam.m[0]["b"].shape))
print("m[1][\"W\"] shape: {}".format(test_adam.m[1]["W"].shape))
print("m[1][\"b\"] shape: {}".format(test_adam.m[1]["b"].shape), "\n")

assert(np.allclose(test_adam.m[0]["W"].shape, np.array([5, 2])))
assert(np.allclose(test_adam.m[0]["b"].shape, np.array([1, 2])))
assert(np.allclose(test_adam.m[1]["W"].shape, np.array([2, 3])))
assert(np.allclose(test_adam.m[1]["b"].shape, np.array([1, 3])))

print("v[0][\"W\"] shape: {}".format(test_adam.v[0]["W"].shape))
print("v[0][\"b\"] shape: {}".format(test_adam.v[0]["b"].shape))
print("v[1][\"W\"] shape: {}".format(test_adam.v[1]["W"].shape))
print("v[1][\"b\"] shape: {}".format(test_adam.v[1]["b"].shape), "\n")

assert(np.allclose(test_adam.v[0]["W"].shape, np.array([5, 2])))
assert(np.allclose(test_adam.v[0]["b"].shape, np.array([1, 2])))
assert(np.allclose(test_adam.v[1]["W"].shape, np.array([2, 3])))
assert(np.allclose(test_adam.v[1]["b"].shape, np.array([1, 3])))

assert(np.all(test_adam.m[0]["W"]==0))
assert(np.all(test_adam.m[0]["b"]==0))
assert(np.all(test_adam.m[1]["W"]==0))
assert(np.all(test_adam.m[1]["b"]==0))

assert(np.all(test_adam.v[0]["W"]==0))
assert(np.all(test_adam.v[0]["b"]==0))
assert(np.all(test_adam.v[1]["W"]==0))
assert(np.all(test_adam.v[1]["b"]==0))

print("Passed the asserts! (Note: These are however limited in scope, additional testing is encouraged.)")


## Test Code for Adam update_weights() ##
network_config = {"state_dim": 5,
                  "num_hidden_units": 2,
                  "num_actions": 3
                 }

optimizer_info = {"step_size": 0.1,
                  "beta_m": 0.99,
                  "beta_v": 0.999,
                  "epsilon": 0.0001
                 }

network = ActionValueNetwork(network_config)
test_adam = Adam(network.layer_sizes, optimizer_info)

rand_generator = np.random.RandomState(0)

# Initialize m and v
test_adam.m[0]["W"] = rand_generator.normal(0, 1, (5, 2))
test_adam.m[0]["b"] = rand_generator.normal(0, 1, (1, 2))
test_adam.m[1]["W"] = rand_generator.normal(0, 1, (2, 3))
test_adam.m[1]["b"] = rand_generator.normal(0, 1, (1, 3))

test_adam.v[0]["W"] = np.abs(rand_generator.normal(0, 1, (5, 2)))
test_adam.v[0]["b"] = np.abs(rand_generator.normal(0, 1, (1, 2)))
test_adam.v[1]["W"] = np.abs(rand_generator.normal(0, 1, (2, 3)))
test_adam.v[1]["b"] = np.abs(rand_generator.normal(0, 1, (1, 3)))

# Specify weights
weights = [dict() for i in range(1, len(test_adam.layer_sizes))]
weights[0]["W"] = rand_generator.normal(0, 1, (5, 2))
weights[0]["b"] = rand_generator.normal(0, 1, (1, 2))
weights[1]["W"] = rand_generator.normal(0, 1, (2, 3))
weights[1]["b"] = rand_generator.normal(0, 1, (1, 3))

# Specify g
g = [dict() for i in range(1, len(test_adam.layer_sizes))]
g[0]["W"] = rand_generator.normal(0, 1, (5, 2))
g[0]["b"] = rand_generator.normal(0, 1, (1, 2))
g[1]["W"] = rand_generator.normal(0, 1, (2, 3))
g[1]["b"] = rand_generator.normal(0, 1, (1, 3))

# Update weights
updated_weights = test_adam.update_weights(weights, g)

# updated weights asserts
updated_weights_answer = np.load("asserts/update_weights.npz")

print("updated_weights[0][\"W\"]\n", updated_weights[0]["W"], "\n")
print("updated_weights[0][\"b\"]\n", updated_weights[0]["b"], "\n")
print("updated_weights[1][\"W\"]\n", updated_weights[1]["W"], "\n")
print("updated_weights[1][\"b\"]\n", updated_weights[1]["b"], "\n")

assert(np.allclose(updated_weights[0]["W"], updated_weights_answer["W0"]))
assert(np.allclose(updated_weights[0]["b"], updated_weights_answer["b0"]))
assert(np.allclose(updated_weights[1]["W"], updated_weights_answer["W1"]))
assert(np.allclose(updated_weights[1]["b"], updated_weights_answer["b1"]))

print("Passed the asserts! (Note: These are however limited in scope, additional testing is encouraged.)")



# Experience Replay Buffers
#
# Experience replay is a simple method that can get some of the advantages of Dyna by saving a buffer of experience 
# and using the data stored in the buffer as a model. This view of prior data as a model works because the data 
# represents actual transitions from the underlying MDP. Furthermore, as a side note, this kind of model that is 
# not learned and simply a collection of experience can be called non-parametric as it can be ever-growing as opposed 
# to a parametric model where the transitions are learned to be represented with a fixed set of parameters or weights.

class ReplayBuffer:
    def __init__(self, size, minibatch_size, seed):
        """
        Args:
            size (integer): The size of the replay buffer.              
            minibatch_size (integer): The sample size.
            seed (integer): The seed for the random number generator. 
        """
        self.buffer = []
        self.minibatch_size = minibatch_size
        self.rand_generator = np.random.RandomState(seed)
        self.max_size = size

    def append(self, state, action, reward, terminal, next_state):
        """
        Args:
            state (Numpy array): The state.              
            action (integer): The action.
            reward (float): The reward.
            terminal (integer): 1 if the next state is a terminal state and 0 otherwise.
            next_state (Numpy array): The next state.           
        """
        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, terminal, next_state])

    def sample(self):
        """
        Returns:
            A list of transition tuples including state, action, reward, terinal, and next_state
        """
        idxs = self.rand_generator.choice(np.arange(len(self.buffer)), size=self.minibatch_size)
        return [self.buffer[idx] for idx in idxs]

    def size(self):
        return len(self.buffer)


# Softmax policy
# 
# One advantage of a softmax policy is that it explores according to the action-values, meaning that an action 
# with a moderate value has a higher chance of getting selected compared to an action with a lower value. 
# Contrast this with an  𝜖 -greedy policy which does not consider the individual action values when choosing an 
# exploratory action in a state and instead chooses randomly when doing so.
#
# Where tau (t) is the temperature parameter which controls how much the agent focuses on the highest valued actions. 
# The smaller the temperature, the more the agent selects the greedy action. Conversely, when the temperature is high, 
# the agent selects among actions more uniformly random.
#
# Given that a softmax policy exponentiates action values, if those values are large, exponentiating them could get very large. 
# To implement the softmax policy in a numerically stable way, we often subtract the maximum action-value from the action-values.

def softmax(action_values, tau=1.0):
    """
    Args:
        action_values (Numpy array): A 2D array of shape (batch_size, num_actions). 
                       The action-values computed by an action-value network.              
        tau (float): The temperature parameter scalar.
    Returns:
        A 2D array of shape (batch_size, num_actions). Where each column is a probability distribution over
        the actions representing the policy.
    """
    # Compute the preferences by dividing the action-values by the temperature parameter tau
    preferences = action_values / tau
    # Compute the maximum preference across the actions
    max_preference = np.max(preferences, axis=1)
    
    # Reshape max_preference array which has shape [Batch,] to [Batch, 1]. This allows NumPy broadcasting 
    # when subtracting the maximum preference from the preference of each action.
    reshaped_max_preference = max_preference.reshape((-1, 1))
    
    # Compute the numerator, i.e., the exponential of the preference - the max preference.
    exp_preferences = np.exp(preferences - reshaped_max_preference)
    # Compute the denominator, i.e., the sum over the numerator along the actions axis.
    sum_of_exp_preferences = np.sum(exp_preferences, axis=1)
    
    # Reshape sum_of_exp_preferences array which has shape [Batch,] to [Batch, 1] to  allow for NumPy broadcasting 
    # when dividing the numerator by the denominator.
    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
    
    # Compute the action probabilities according to the equation in the previous cell.
    action_probs = exp_preferences / reshaped_sum_of_exp_preferences
    
    # squeeze() removes any singleton dimensions. It is used here because this function is used in the 
    # agent policy when selecting an action (for which the batch dimension is 1.) As np.random.choice is used in 
    # the agent policy and it expects 1D arrays, we need to remove this singleton batch dimension.
    action_probs = action_probs.squeeze()
    return action_probs


## Test Code for softmax() ##
#
# NOTE: The test below is limited in scope. Additional tests are used in the autograder, so it is recommended 
# to test your implementations more carefully for correctness.
rand_generator = np.random.RandomState(0)
action_values = rand_generator.normal(0, 1, (2, 4))
tau = 0.5

action_probs = softmax(action_values, tau)
print("action_probs", action_probs)

assert(np.allclose(action_probs, np.array([
    [0.25849645, 0.01689625, 0.05374514, 0.67086216],
    [0.84699852, 0.00286345, 0.13520063, 0.01493741]
])))

print("Passed the asserts! (Note: These are however limited in scope, additional testing is encouraged.)")



### RL_Glue Agent ###
#
# The main component that you will implement is the action-value network updates with experience sampled 
# from the experience replay buffer.
#
# At time t , we have an action-value function represented as a neural network, say Q_t. We want to update our 
# action-value function and get a new one we can use at the next timestep. We will get this  Q_t+1  using multiple 
# replay steps that each result in an intermediate action-value function  Qi_t+1  where  i  indexes which replay step we are at.
# 
# In each replay step, we sample a batch of experiences from the replay buffer and compute a minibatch Expected-SARSA update. 
# Across these N replay steps, we will use the current "un-updated" action-value network at time  t ,  Q_t , for computing 
# the action-values of the next-states. This contrasts using the most recent action-values from the last replay step  Qi_t+1 . 
# We make this choice to have targets that are stable across replay steps.

def get_td_error(states, next_states, actions, rewards, discount, terminals, network, current_q, tau):
    """
    Args:
        states (Numpy array): The batch of states with the shape (batch_size, state_dim).
        next_states (Numpy array): The batch of next states with the shape (batch_size, state_dim).
        actions (Numpy array): The batch of actions with the shape (batch_size,).
        rewards (Numpy array): The batch of rewards with the shape (batch_size,).
        discount (float): The discount factor.
        terminals (Numpy array): The batch of terminals with the shape (batch_size,).
        network (ActionValueNetwork): The latest state of the network that is getting replay updates.
        current_q (ActionValueNetwork): The fixed network used for computing the targets, 
                                        and particularly, the action-values at the next-states.
    Returns:
        The TD errors (Numpy array) for actions taken, of shape (batch_size,)
    """
    
    # Note: Here network is the latest state of the network that is getting replay updates. In other words, 
    # the network represents Q_{t+1}^{i} whereas current_q represents Q_t, the fixed network used for computing the 
    # targets, and particularly, the action-values at the next-states.
    
    # Compute action values at next states using current_q network
    # Note that q_next_mat is a 2D array of shape (batch_size, num_actions)    
    q_next_mat = current_q.get_action_values(next_states)
    
    # Compute policy at next state by passing the action-values in q_next_mat to softmax()
    # Note that probs_mat is a 2D array of shape (batch_size, num_actions)
    probs_mat = softmax(q_next_mat,tau)
    
    # Compute the estimate of the next state value, v_next_vec.
    # Hint: sum the action-values for the next_states weighted by the policy, probs_mat. Then, multiply by
    # (1 - terminals) to make sure v_next_vec is zero for terminal next states.
    # Note that v_next_vec is a 1D array of shape (batch_size,)
    weighted_next_mat = q_next_mat * probs_mat
    action_values_sum = np.sum(weighted_next_mat, axis=1)
    v_next_vec = action_values_sum * (1-terminals)
    
    # Compute Expected Sarsa target
    # Note that target_vec is a 1D array of shape (batch_size,)
    target_vec = rewards + (discount * v_next_vec)
    
    # Compute action values at the current states for all actions using network
    # Note that q_mat is a 2D array of shape (batch_size, num_actions)
    q_mat = network.get_action_values(states)
    
    # Batch Indices is an array from 0 to the batch size - 1. 
    batch_indices = np.arange(q_mat.shape[0])

    # Compute q_vec by selecting q(s, a) from q_mat for taken actions
    # Use batch_indices as the index for the first dimension of q_mat
    # Note that q_vec is a 1D array of shape (batch_size)
    q_vec = q_mat[batch_indices, actions]
    
    # Compute TD errors for actions taken
    # Note that delta_vec is a 1D array of shape (batch_size)
    delta_vec = target_vec - q_vec
    
    return delta_vec


## Test Code for get_td_error() ##
#
# NOTE: The test below is limited in scope. Additional tests are used in the autograder, so it is recommended 
# to test your implementations more carefully for correctness.

data = np.load("asserts/get_td_error_1.npz", allow_pickle=True)

states = data["states"]
next_states = data["next_states"]
actions = data["actions"]
rewards = data["rewards"]
discount = data["discount"]
terminals = data["terminals"]
tau = 0.001

network_config = {"state_dim": 8,
                  "num_hidden_units": 512,
                  "num_actions": 4
                  }

network = ActionValueNetwork(network_config)
network.set_weights(data["network_weights"])

current_q = ActionValueNetwork(network_config)
current_q.set_weights(data["current_q_weights"])

delta_vec = get_td_error(states, next_states, actions, rewards, discount, terminals, network, current_q, tau)
answer_delta_vec = data["delta_vec"]

assert(np.allclose(delta_vec, answer_delta_vec))
print("Passed the asserts! (Note: These are however limited in scope, additional testing is encouraged.)")


