"""
Main Script for QNN using Q-Learning.
"""

# General imports
import numpy as np

# Qiskit Circuit imports
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.library import TwoLocal

# Qiskit imports
import qiskit as qk
import qiskit_aer

# Qiskit Machine Learning imports
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector

# PyTorch imports
import torch
from torch import Tensor
from torch.optim import LBFGS, SGD, Adam, RMSprop

import rep_env

# OpenAI Gym import
from numpy import int32

# Fix seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

import helper as hlp

# Wandb Logging
# Keep it False for now.
wandb_logging = True

if wandb_logging:
    # Logging Package
    import wandb

# Random Seed for Reproducibility
rng = np.random.default_rng(seed=12345)


# Functions to make the PQC Circuit.
def encoding_circuit(inputs, num_qubits=4, *args):
    qc = qk.QuantumCircuit(num_qubits)

    # Encode data with a RX rotation
    for i in range(len(inputs)):
        qc.rx(inputs[i], i)

    return qc


def parametrized_circuit(num_qubits=4, reuploading=False, reps=2, insert_barriers=True, meas=False):
    qr = qk.QuantumRegister(num_qubits, 'qr')
    qc = qk.QuantumCircuit(qr)

    if meas:
        qr = qk.QuantumRegister(num_qubits, 'qr')
        cr = qk.ClassicalRegister(num_qubits, 'cr')
        qc = qk.QuantumCircuit(qr, cr)

    if not reuploading:

        # Define a vector containg Inputs as parameters (*not* to be optimized)
        inputs = qk.circuit.ParameterVector('x', num_qubits)

        # Encode classical input data
        qc.compose(encoding_circuit(inputs, num_qubits=num_qubits), inplace=True)
        if insert_barriers:
            qc.barrier()

        # Variational circuit
        qc.compose(TwoLocal(num_qubits, ['ry', 'rz'], 'cz', 'circular',
                            reps=reps, insert_barriers=insert_barriers,
                            skip_final_rotation_layer=True), inplace=True)
        if insert_barriers:
            qc.barrier()

        # Add final measurements
        if meas:
            qc.measure(qr, cr)

    elif reuploading:

        # Define a vector containing Inputs as parameters (*not* to be optimized)
        inputs = qk.circuit.ParameterVector('x', num_qubits)

        # Define a vector containing variational parameters
        θ = qk.circuit.ParameterVector('θ', 2 * num_qubits * reps)

        # Iterate for a number of repetitions
        for rep in range(reps):

            # Encode classical input data
            qc.compose(encoding_circuit(inputs, num_qubits=num_qubits), inplace=True)
            if insert_barriers: qc.barrier()

            # Variational circuit (does the same as TwoLocal from Qiskit)
            for qubit in range(num_qubits):
                qc.ry(θ[qubit + 2 * num_qubits * (rep)], qubit)
                qc.rz(θ[qubit + 2 * num_qubits * (rep) + num_qubits], qubit)
            if insert_barriers: qc.barrier()

            # Add entanglers (this code is for a circular entangler)
            qc.cz(qr[-1], qr[0])
            for qubit in range(num_qubits - 1):
                qc.cz(qr[qubit], qr[qubit + 1])
            if insert_barriers: qc.barrier()

        # Add final measurements
        if meas: qc.measure(qr, cr)

    return qc


# ### Create the PQC
# We can use the functions just defined to create the Parametrized Quantum Circuit:

# Select the number of qubits
num_qubits = 3
layers = 6
reuploading = True

# Generate the Parametrized Quantum Circuit (note the flags reuploading and reps)
qc = parametrized_circuit(num_qubits=num_qubits,
                          reuploading=reuploading,
                          reps=layers)

# Fetch the parameters from the circuit and divide them in Inputs (X) and Trainable Parameters (params)
# The first four parameters are for the inputs 
X = list(qc.parameters)[: num_qubits]

# The remaining ones are the trainable weights of the quantum neural network
params = list(qc.parameters)[num_qubits:]

qc.draw()

# Construct an ideal simulator
aersim = qiskit_aer.AerSimulator()

# Create a Quantum Neural Network object starting from the quantum circuit defined above
qnn = CircuitQNN(qc, input_params=X, weight_params=params,
                 quantum_instance=aersim)

# Connect to PyTorch
initial_weights = (2 * np.random.rand(qnn.num_weights) - 1)
quantum_nn = TorchConnector(qnn, initial_weights)


# PyTorch Layers for pre-processing and post-processing.
class encoding_layer(torch.nn.Module):
    def __init__(self, num_qubits=4):
        super().__init__()

        # Define weights for the layer
        weights = torch.Tensor(num_qubits)
        self.weights = torch.nn.Parameter(weights)
        torch.nn.init.uniform_(self.weights, -1, 1)  # <--  Initialization strategy

    def forward(self, x):
        """Forward step, as explained above."""

        if not isinstance(x, Tensor):
            x = Tensor(x)

        x = self.weights * x
        x = torch.atan(x)

        return x


class exp_val_layer(torch.nn.Module):
    def __init__(self, action_space=2):
        super().__init__()

        # Define the weights for the layer
        weights = torch.Tensor(action_space)
        self.weights = torch.nn.Parameter(weights)
        torch.nn.init.uniform_(self.weights, 35, 40)  # <-- Initialization strategy (heuristic choice)

        self.mask_Z0 = torch.tensor([-1., 1., -1., 1., -1., 1., -1., 1.], requires_grad=False)
        self.mask_Z1 = torch.tensor([-1., -1., 1., 1., -1., -1., 1., 1.], requires_grad=False)
        self.mask_Z2 = torch.tensor([-1., -1., -1., -1., 1., 1., 1., 1.], requires_grad=False)
        self.mask_Z012 = torch.tensor([-1., 1., 1., -1., 1., -1., -1., 1.], requires_grad=False)

    def forward(self, x):
        """Forward step, as described above."""
        expval_Z0 = self.mask_Z0 * x
        expval_Z1 = self.mask_Z1 * x
        expval_Z2 = self.mask_Z2 * x
        expval_Z012 = self.mask_Z012 * x

        # Single sample
        if len(x.shape) == 1:
            expval_Z0 = torch.sum(expval_Z0)
            expval_Z1 = torch.sum(expval_Z1)
            expval_Z2 = torch.sum(expval_Z2)
            expval_Z012 = torch.sum(expval_Z012)
            out = torch.cat(
                (expval_Z0.unsqueeze(0), expval_Z1.unsqueeze(0), expval_Z2.unsqueeze(0), expval_Z012.unsqueeze(0)))

        # Batch of samples
        else:
            expval_Z0 = torch.sum(expval_Z0, dim=1, keepdim=True)
            expval_Z1 = torch.sum(expval_Z1, dim=1, keepdim=True)
            expval_Z2 = torch.sum(expval_Z2, dim=1, keepdim=True)
            expval_Z012 = torch.sum(expval_Z012, dim=1, keepdim=True)
            out = torch.cat((expval_Z0, expval_Z1, expval_Z2, expval_Z012), 1)

        return self.weights * ((out + 1.) / 2.)


# Classical trainable preprocessing
encoding = encoding_layer(num_qubits=3)

# Classical trainable postprocessing
exp_val = exp_val_layer(action_space=4)

# Stack the classical and quantum layers together 
model = torch.nn.Sequential(encoding,
                            quantum_nn,
                            exp_val)
# model.state_dict()

# Defining the environment.
Starting_Capital = 1000
env = rep_env.rep_env(ProductID=4, StoreID=14, Capital=Starting_Capital)


input_shape = [3]  # == env.observation_space.shape
n_outputs = 4  # == env.action_space.n

# Training Functions.
from collections import deque

replay_memory = deque(maxlen=2000)


def epsilon_greedy_policy(state, epsilon=0):
    """Manages the transition from the *exploration* to *exploitation* phase"""
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        with torch.no_grad():
            Q_values = model(Tensor(state)).numpy()
            # Q_values = model(state).detach().numpy()
        # return np.argmax(Q_values[0])
        return np.argmax(Q_values)


def sample_experiences(batch_size):
    """Sample some experiences from the replay memory"""
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]

    states = np.array([experience[0] for experience in batch])
    actions = np.array([experience[1] for experience in batch])
    rewards = np.array([experience[2] for experience in batch])
    next_states = np.array([experience[3] for experience in batch])
    dones = np.array([experience[4] for experience in batch])

    return states, actions, rewards, next_states, dones


def play_one_step(env, state, epsilon):
    """Perform one action in the environment and register the state of the system"""
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, _, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info


def sequential_training_step(batch_size):
    """
    Actual training routine. Implements the Deep Q-Learning algorithm.
    """

    # Sample past experiences 
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences

    # Evaluates Target Q-values
    with torch.no_grad():
        next_Q_values = model(Tensor(next_states)).numpy()
    max_next_Q_values = np.max(a=next_Q_values, axis=1)
    target_Q_values = (rewards + (1 - dones) * discount_rate * max_next_Q_values)

    # Accumulate Loss sequentially (if batching data, gradients of the parameters are vanishing)
    loss = 0.
    for j, state in enumerate(states):
        single_Q_value = model(Tensor(state))
        Q_value = single_Q_value[actions[j]]
        loss += (target_Q_values[j] - Q_value) ** 2
    # Evaluate the gradients and update the parameters 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Hyperparameters.
batch_size = 4
discount_rate = 0.99
learning_rate = 1e-2
optimizer = Adam(model.parameters(), lr=learning_rate)

rewards = []
best_score = 0
episodes = 100
max_steps = env.n_weeks + 1
exploration_ratio = 0.9
exploration_episode = round(exploration_ratio * episodes)
exploration_experience = 40

hyperparameters = {
    "Starting Capital": Starting_Capital,
    "num_qubits": num_qubits,
    "layers": layers,
    "reuploading": reuploading,
    "batch_size": batch_size,
    "discount_rate": discount_rate,
    "learning_rate": learning_rate,
    "episodes": episodes,
    "exploration_ratio": exploration_ratio,
    "exploration_experience": exploration_experience
}

if wandb_logging:
    wandb.init(project='tcs-bloq-qrl', config=hyperparameters, tags='QNN')

print(f'The weekly demand is: {env.weekly_demand}')

# Main Training Loop

from tqdm import tqdm

for episode in tqdm(range(episodes), colour='green', miniters=1):
    # wandb.watch(model)

    # Run environment simulation
    obs = env.reset()[0]
    ep_reward = 0

    for step in range(max_steps):
        # Manages the transition from exploration to exploitation
        epsilon = max(1 - episode / exploration_episode, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        ep_reward += reward
        if done:
            break
    rewards.append(ep_reward)

    # Saving best agent
    if ep_reward >= best_score:
        best_score = ep_reward

    if wandb_logging:
        wandb.log({'Train Reward': ep_reward})

    if episode % 100 == 0:
        print(f'Episode: {episode},Reward: {ep_reward}')

    # Start training only after some exploration experiences  
    if episode >= exploration_experience:
        sequential_training_step(batch_size)

print(f'Best score in Training: {best_score}')

if wandb_logging:
    wandb.log({'Best Training score': best_score})

torch.save(model.state_dict(),
           "models/" + hlp.generate_filename(hyperparameters, file_type="Model", model_type='qnn') + '.pth')


def plot_rewards(rewards, save):
    plt.figure(figsize=(17, 5))
    plt.plot(rewards, lw=1, label='Training Rewards')

    plt.title("Training QNN Rewards", fontsize=14)

    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.grid()
    plt.legend()

    plot_fname = hlp.generate_filename(hyperparameters=hyperparameters, file_type='Training_Rewards', model_type='qnn')
    plot_path = 'plots/' + plot_fname + '.png'
    if save:
        plt.savefig(plot_path)

    if wandb_logging:
        wandb.log({'Training Rewards Plot': wandb.Image(plot_path)})
    # plt.show()


# Plotting the training rewards
plot_rewards(rewards, save=True)


# Testing our model.
def test(test_evals=25, save=True):
    eval_rewards = []
    for _ in range(test_evals):
        rewards = 0
        state = torch.Tensor(env.reset()[0])
        print(f'Initial State:\n{state[2]} {state[0]} {state[1]}')

        print(f'\nDemand Capital On_Hand Profit Replenishment Predicted Sale Actual Sale')
        for step in range(env.n_weeks+1):
            action = epsilon_greedy_policy(state)
            state, reward, done, _, info = env.step(action)
            rewards += reward
            if not done:
                print(
                    f'{state[2]} {round(state[0], 2)} {state[1]} {round(reward, 2)} {action} {info["Sampled Sale"]} {info["Sale"]}')
            else:
                print(f'{state[2]} {round(state[0], 2)} {state[1]} {round(reward, 2)} {action}\n{info["msg"]}')
                print(f'Total Reward: {rewards}\n')
        eval_rewards.append(rewards)
        if wandb_logging:
            wandb.log({'Test Reward': rewards})
    print(f'Average Test Reward: {np.mean(eval_rewards)}')

    plt.figure(figsize=(15, 5))
    plt.plot(range(test_evals), eval_rewards)
    plt.grid()

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Testing')

    # plt.text(1, round(average_eval_reward) - 10, f"Average Reward: {round(average_eval_reward,2)}", fontsize=12)
    plot_fname = hlp.generate_filename(hyperparameters, file_type='Testing Rewards', model_type='qnn')
    plot_path = 'plots/' + plot_fname + '.png'
    if save:
        plt.savefig(plot_path)
    # plt.show()

    if wandb_logging:
        wandb.log({'Average Test Reward': np.mean(eval_rewards)})
        wandb.log({'Testing Rewards Plot': wandb.Image(plot_path)})


# Testing our model
test(save=True)

# if wandb_logging:
#     wandb.finish()
