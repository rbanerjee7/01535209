# Classical RL Testing.

# Importing Libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import helper as hlp

import tensorflow as tf
import rep_env


# Logging.
# wandb logging. Keep it False for now.
wandb_logging = False

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

if wandb_logging:
    wandb.init(project='tcs-bloq-qrl', config={},tags='classical RL')

print(f'The weekly demand is: {env.weekly_demand}')

# Calling the agent.
dqn_agent(np.array([1,2,2])[np.newaxis, :]);

# Loading a pre-trained model.
dqn_agent.load_weights('models\Model_cnn_0.99_0.001_1000_0.7_0.995_t_15_05_d_08_03_2024.keras')

max_steps_per_episode = env.n_weeks + 1

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
				print(f'{	state[2]} {round(state[0],2)} {state[1]} {round(reward,2)} {action}\n{info["msg"]}')
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
	plot_fname = hlp.generate_filename(hyperparameters={'cnn':'Testing'},file_type='Testing_Rewards',model_type='cnn')
	plot_path = 'plots/' + plot_fname + '.png'

	plt.savefig(plot_path) 

	if wandb_logging:
		wandb.log({'Testing Rewards Plot':wandb.Image(plot_path)})

test(num_eval_episodes=100,save=True)

# Model Summary
# dqn_agent.summary()

# Finishing wandb logging.
if wandb_logging:
    wandb.finish()