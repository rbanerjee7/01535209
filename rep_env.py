"""
Script for custom environment.
"""

# General imports
import numpy as np
# OpenAI Gym import
from numpy import int32
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import helper as hlp


# forecast = pd.read_csv('data_edit/FORECAST.csv', parse_dates=['date'], date_format='%d-%m-%Y')  # Forecast
store_stock = pd.read_csv('data_edit/STORE_STOCK.csv', parse_dates=['date'], date_format='%d-%m-%Y')  # Store Stock
products = pd.read_csv('data_edit/PRODUCTS.csv')

# Random Seed for Reproducibility
rng = np.random.default_rng(seed=12345)


# Function to return the actual sales which is sampled from a distribution.
def actual_sale(demand: int):
    """
    Predict sale based on the demand using a probability distribution.

    Parameters:
    demand (int): The demand for the product.

    Returns:
    int: Sale Predicted based on the demand.
    """
    sale_prob = np.arange(demand + 1)
    sale_prob = (sale_prob + 1) ** 2  # (x+1)^2 probability distribution
    sale_prob = sale_prob / sale_prob.sum()
    return rng.choice(np.arange(demand + 1), size=1, p=sale_prob)[0]


class rep_env(gym.Env):
    """A retail replenishment environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, ProductID: int, StoreID: int, Capital: int):

        # Product and Store Details
        self.ProductID = ProductID
        self.StoreID = StoreID

        self.available_capital = Capital
        self.init_capital = Capital
        self.week = 0
        self.n_weeks = 25 # Number of weeks the environment will run for.

        # self.weekly_demand = hlp.filter_product_store(self.ProductID, self.StoreID, forecast)['fcst'].to_numpy()
        # self.weekly_demand = rng.choice([0, 1, 2, 3], size=self.n_weeks+1)
        # self.weekly_demand = rng.choice([1, 2, 3], size=self.n_weeks+1)
        self.weekly_demand = rng.choice([2, 3], size=self.n_weeks+1)
        self.on_hand = hlp.filter_product_store(self.ProductID, self.StoreID, store_stock)['oh'].to_numpy()[0]

        self.demand = self.weekly_demand[self.week]

        self.cost = hlp.filter_product(2, products, sort=False)['Cost Price'].to_numpy()[0]
        self.holding_cost = hlp.filter_product(2, products, sort=False)['Holding Cost'].to_numpy()[0]
        self.profit = hlp.filter_product(2, products, sort=False)['Profit'].to_numpy()[0]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "Capital": spaces.Box(low=0, high=1e7, shape=(1,)),
                "On Hand": spaces.Box(low=0, high=1e2, shape=(1,), dtype=int32),
                "Forecasted Demand": spaces.Box(low=0, high=1e2, shape=(1,), dtype=int32),
            }
        )

    def _get_obs(self):
        return np.array([self.available_capital, self.on_hand, self.demand])

    def step(self, action):
        # Output format
        # (observation, reward, terminated, truncated, info)

        # Actual Sale
        sampled_sale = actual_sale(self.demand)
        sale = int(min(sampled_sale, action + self.on_hand))

        # Termination variable for an episode. 
        # False-> not done 
        # True-> done
        done = False

        # Action/Replenishment cannot be greater than demand.
        # if action > self.demand:
        #     done = True
        #     return self._get_obs(), 0, done, False, {"msg": 'Episode Terminated. Replenishment is greater than demand.'}

        # Time Period Exceeded
        if self.week >= self.n_weeks:
            done = True
            return self._get_obs(), 0, done, False, {"msg": f'Episode Finished after {self.n_weeks+1} weeks'}

        # Calculating the weekly cost and revenue 
        weekly_cost = action * (self.cost + self.holding_cost) + max((action - self.demand), 0) * self.holding_cost
        # weekly_revenue = sale * (self.cost + self.profit) - max((self.demand - action), 0) * (self.profit - self.holding_cost)
        weekly_revenue = sale * (self.cost + self.profit)

        # Capital Constraint 
        if weekly_cost > self.available_capital:
            done = True
            return self._get_obs(), 0, done, False, {"msg": 'Episode Terminated. Insufficient Capital'}

        # Profit 
        profit = weekly_revenue - weekly_cost

        # Calculating updated capital and inventory 
        updated_capital = self.available_capital + profit
        updated_on_hand = self.on_hand + action - sale

        # Updating the state
        self.available_capital = updated_capital
        self.on_hand = updated_on_hand
        self.week += 1
        self.demand = self.weekly_demand[self.week]

        # (observation, reward, terminated, truncated, info)
        return self._get_obs(), profit, done, False, {'Sampled Sale': sampled_sale, 'Sale': sale}

    # Resetting the environment
    def reset(self, seed=1):

        self.on_hand = hlp.filter_product_store(self.ProductID, self.StoreID, store_stock)['oh'].to_numpy()[0]
        self.available_capital = self.init_capital
        self.week = 0

        return self._get_obs(), {}

    def render(self, mode='human', close=False):
        # Render the environment to the screen  
        print(f'Step: {self.week}')
        print(f'Capital: {self.available_capital}')
        print(f'OH: {self.on_hand}')

    def __repr__(self) -> str:
        return f'ProductID: {self.ProductID}\nStoreID: {self.StoreID}\nCost: {self.cost}\nHolding Cost: {self.holding_cost}\nProfit: {self.profit}\nDemand: {self.demand}\nOn_hand: {self.on_hand}\nAvailable Capital: {self.available_capital}\nWeek: {self.week}'

    def __str__(self):
        return f'ProductID: {self.ProductID}\nStoreID: {self.StoreID}\nCost: {self.cost}\nHolding Cost: {self.holding_cost}\nProfit: {self.profit}\nDemand: {self.demand}\nOn_hand: {self.on_hand}\nAvailable Capital: {self.available_capital}\nWeek: {self.week}'
