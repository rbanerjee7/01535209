# Classical RL Training and Testing.

# Importing Libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

import gymnasium as gym
from gymnasium import spaces

import helper as hlp

import tensorflow as tf
import gymnasium as gym
import rep_env

from tqdm import tqdm

# Logging.
# wandb logging. Keep it False for now.
wandb_logging = True

if wandb_logging:
    # Logging Package
    import wandb

# Random Seed for Reproducibility.
rng = np.random.default_rng(seed=12345)

# Define the DQN Model.
class DQN(tf.keras.Model):
	def __init__(self, num_actions):
		super(DQN, self).__init__()
		self.dense1 = tf.keras.layers.Dense(24, activation='relu')
		self.dense2 = tf.keras.layers.Dense(24, activation='relu')
		self.output_layer = tf.keras.layers.Dense(
			num_actions, activation='linear')

	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.dense2(x)
		return self.output_layer(x)

num_actions = 4
dqn_agent = DQN(num_actions)

# Define the DQN Algorithm Parameters.
learning_rate = 0.001
discount_factor = 0.99
# Initial exploration probability
# exploration_prob = 1.0
exploration_prob = 0.7
# Decay rate of exploration probability
exploration_decay = 0.995
# Minimum exploration probability
min_exploration_prob = 0.01

# Define the Loss Function and Optimizer.
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Defining the environment.
Starting_Capital = 1000
env = rep_env.rep_env(ProductID=4, StoreID=14, Capital=Starting_Capital)


# Training Parameters.
num_episodes = 1000
max_steps_per_episode = env.n_weeks + 1
rewards = np.zeros(num_episodes)
best_score = 0

hyperparameters = {

    "discount_rate": discount_factor,
    "learning_rate": learning_rate,
    "episodes": num_episodes,
	"exploration_prob": exploration_prob,
	"exploration_decay": exploration_decay,

}
if wandb_logging:
    wandb.init(project='tcs-bloq-qrl', config=hyperparameters,tags='classical RL')

print(f'The weekly demand is: {env.weekly_demand}')

# Training the DQN.
for episode in tqdm(range(num_episodes),colour='green', miniters=1):
	state,_ = env.reset()
	episode_reward = 0

	for step in range(max_steps_per_episode):
		# Choose action using epsilon-greedy policy
		if np.random.rand() < exploration_prob:
			action = env.action_space.sample() # Explore randomly
		else:
			action = np.argmax(dqn_agent(state[np.newaxis, :]))

		next_state, reward, done, _, __ = env.step(action)

		# Update the Q-values using Bellman equation
		with tf.GradientTape() as tape:
			current_q_values = dqn_agent(state[np.newaxis, :])
			next_q_values = dqn_agent(next_state[np.newaxis, :])
			max_next_q = tf.reduce_max(next_q_values, axis=-1)
			target_q_values = current_q_values.numpy()
			target_q_values[0, action] = reward + discount_factor * max_next_q * (1 - done)
			loss = loss_fn(current_q_values, target_q_values)

		gradients = tape.gradient(loss, dqn_agent.trainable_variables)
		optimizer.apply_gradients(zip(gradients, dqn_agent.trainable_variables))

		state = next_state
		episode_reward += reward

		if done:
			rewards[episode] = episode_reward
			if episode-1 % 100 == 0:
				print(f"Episode {episode + 1}: Reward = {round(episode_reward)}")
			break
	if episode_reward >= best_score:
		best_score = episode_reward
	
	if wandb_logging:
		wandb.log({'Train Reward': episode_reward})

	# Decay exploration probability
	exploration_prob = max(min_exploration_prob, exploration_prob * exploration_decay)

if wandb_logging:
	wandb.log({'Best score in Training': best_score})

model_fname = hlp.generate_filename(hyperparameters=hyperparameters,file_type='Model',model_type='cnn')

dqn_agent.save("models/" + model_fname + '.keras')

# Plottting the Training
def plot_rewards(rewards,save):
    plt.figure(figsize=(17,5))
    plt.plot(np.arange(num_episodes), rewards, 'g', label='Training Rewards',lw=1)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.title('Training CNN Rewards', fontsize=14)

    # plt.ylim(-100, 600)

    plt.legend()
    plt.grid()

    plot_fname = hlp.generate_filename(hyperparameters=hyperparameters,file_type='Training_Rewards',model_type='cnn')
    plot_path = 'plots/' + plot_fname + '.png'
    
    if save:
        plt.savefig(plot_path)

    img = wandb.Image('plots/' + plot_fname + '.png')
    wandb.log({'Training Rewards Plot':img})
    # plt.show()

plot_rewards(rewards,save=True)

# Testing the Trained model
def test(num_eval_episodes,save):
	eval_rewards = []

	for _ in range(num_eval_episodes):
		state, _  = env.reset()
		eval_reward = 0
		
		print(f'Initial State:\n{state[2]} {state[0]} {state[1]}')

		print(f'\n\nDemand Capital On_Hand Profit Replenishment Predicted Sale Actual Sale')

		for _ in range(max_steps_per_episode):
		
			action = np.argmax(dqn_agent(state[np.newaxis, :]))
			next_state, reward, done, _, info = env.step(action)

			eval_reward += reward
			state = next_state

			if not done:
				print(f'{state[2]} {round(state[0],2)} {state[1]} {round(reward,2)} {action} {info["Sampled Sale"]} {info["Sale"]}')
			else:
				print(f'{state[2]} {round(state[0],2)} {state[1]} {round(reward,2)} {action}\n{info["msg"]}')
				print(f'\nTotal Reward: {eval_reward}')

			if done:
				break

		eval_rewards.append(eval_reward)
		if wandb_logging:
			wandb.log({'Test Reward': eval_reward})

	average_eval_reward = np.mean(eval_rewards)
	print(f"Average Evaluation Reward: {average_eval_reward}")
	if wandb_logging:
		wandb.log({'Average Test Reward': average_eval_reward})


	# Plottting the Testing Results.
	plt.figure(figsize=(15,5))
	plt.plot(range(num_eval_episodes),eval_rewards)
	plt.grid()

	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.title('Testing')

	# plt.text(1, round(average_eval_reward) - 10, f"Average Reward: {round(average_eval_reward,2)}", fontsize=12)
	plot_fname = hlp.generate_filename(hyperparameters=hyperparameters,file_type='Testing_Rewards',model_type='cnn')
	plot_path = 'plots/' + plot_fname + '.png'

	plt.savefig(plot_path) 

	if wandb_logging:
		wandb.log({'Testing Rewards Plot':wandb.Image(plot_path)})

	# plt.show()

test(num_eval_episodes=100,save=True)

# Model Summary
# dqn_agent.summary()

# Finishing wandb logging.
if wandb_logging:
    wandb.finish()

# %%



