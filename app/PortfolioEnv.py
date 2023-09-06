from __future__ import annotations

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from PortfolioClasses import *
from Data import *
    
class PortfolioEnv(gym.Env):
    """Custom Portfolio Environment that follows gym interface"""

    def __init__(self, df, total_return_wanted, volatility_wanted, prices_df, prices_bmk, starting_holdings, simulation_dates, portfolio_manager_behaviour,  state_space, action_space):
    # def __init__(self, df, total_return_wanted, volatility_wanted, prices_df, prices_bmk, starting_holdings, simulation_dates, portfolio_manager_behaviour, change_behaviour = False, make_plots = False, state_space, action_space, previous_state = [], model_name = "", mode = "", iteration = "", ):
        # simulation days
        self.simulation_days = 300
        self.simulation_dates = simulation_dates
        self.nb_days = len(self.simulation_dates[:self.simulation_days-1])
        self.day = 0

        # the input data
        self.df = df
        self.prices_df = prices_df
        self.prices_bmk = prices_bmk
        self.starting_holdings = starting_holdings
        self.end_date = end_date=simulation_dates.max()
        self.portfolio_manager_behaviour = portfolio_manager_behaviour
        self.nav_data = None
        self.perf_data = None

        # for the model
        self.state_space = state_space
        self.action_space = action_space
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.terminal = False
        self.make_plots = False
        self.change_behaviour = False
        # self.print_verbosity = print_verbosity
        # self.previous_state = previous_state
        # self.model_name = model_name
        # self.mode = mode

        # initialize reward
        self.total_return_wanted = total_return_wanted
        self.volatility_wanted = volatility_wanted
        
        self.reward = 0
        self.episode = 0
        
        # self.total_return = 0
        # self.volatility = 0
        # self.maxdrawdown = 0
        # self.IR = 0
        # self.hit_ratio = 0
        # self.win_loss_ratio = 0
        # self.portfolio = None

        # memorize all the total balance change
        self.rewards_memory = []
        self.actions_memory = []
        # self.date_memory = [self._get_date()]


    def step(self, action):
        
        # Run the simulation loop
        if self.day < self.nb_days:
            self.done = False
        else:
            self.done = True
        
        self.day += 1
        self.portfolio.step()
        portfolio_copy = copy.deepcopy(self.portfolio)

        # Get the updated data for calculating the performances 
        data = portfolio_copy.datacollector.get_agent_vars_dataframe()
        
        # Get the performances data to generate the reward
        self.total_return, self.volatility, self.maxdrawdown, self.IR, self.hit_ratio, self.win_loss_ratio = self._get_performance(portfolio_copy,data)

        #Calculate the reward
        self.reward = self._get_reward()
        self.observation = [self.total_return, self.volatility, self.maxdrawdown, self.IR, self.hit_ratio, self.win_loss_ratio]


        info = {}


        return self.observation, self.reward, self.done , info


    def reset(self):

        # initiate state
        self.portfolio = Portfolio(1, self.starting_holdings, self.portfolio_manager_behaviour, self.simulation_dates, self.prices_df, self.prices_bmk)
        self.reward = 0
        self.day = 0
        self.total_return = 0
        self.volatility = 0
        self.maxdrawdown = 0
        self.IR = 0
        self.hit_ratio = 0
        self.win_loss_ratio = 0
        self.done = False

        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        # self.date_memory = [self._get_date()]

        self.observation = [self.total_return, self.volatility, self.maxdrawdown, self.IR, self.hit_ratio, self.win_loss_ratio]

        self.episode += 1

        return self.observation

    
    # def render(self, mode='human'):
    #     ...

    # def close (self):
    #     ...

    # def _get_date(self):

    #     if len(self.df.tic.unique()) > 1:
    #         date = self.data.date.unique()[0]
    #     else:
    #         date = self.data.date
    #     return date


    def calculateMetrics(data, percentBoolean):
        avg = mean(data)
        med = median(data)
        minimum = min(data)
        maximum = max(data)
        if percentBoolean:
            return toPercentAndRound([minimum, med, avg, maximum], True)
        return toPercentAndRound([minimum, med, avg, maximum], False)


    def toPercentAndRound(nums, percentBoolean):
        if percentBoolean:
            for i in range(len(nums)):
                nums[i] = round(nums[i] * 100, 2)
        else:
            for i in range(len(nums)):
                nums[i] = round(nums[i], 2)
        return nums


    def formatPercentages(nums):
        for i in range(len(nums)):
                nums[i] = f"{nums[i]}%"
        return nums


    def _get_performance(self, portfolio_copy, data):

        self.nav_data = pd.pivot_table(data[~data['date'].isna()], index=['date'], columns=data.index.get_level_values('AgentID'), values='NAV')
        self.perf_data = data[~data['date'].isna()]
        perf = pd.pivot_table(self.perf_data, index=['date'], columns=self.perf_data.index.get_level_values('AgentID'), values='performance')
        gg_perf = perf.diff(-1).agg(['count','sum','std', IR, MaxDrawdown, Volatility]).transpose()

        most_recent_date = self.perf_data['date'].max()
        most_recent_perf = perf[perf.index == most_recent_date]
        totalReturn = most_recent_perf.iloc[-1]

        all_ptfs = [obj for obj in portfolio_copy.schedule.agents if ((isinstance(obj, PortfolioManager)))]
        cols=["pft_id","number_bets", 'nav', "performance", "hit_ratio","win_loss_ratio"]
        behave=pd.DataFrame([],columns=cols)

        hitRatioData = []
        winLossRatioData = []
        i=0

        for ptf in all_ptfs:
            bets=ptf.bets['closed'].copy()
            still_active_bets=ptf.bets['active']
            still_active_bets['security_id']=ptf.bets['active'].index
            still_active_bets['end_date']=end_date
            still_active_bets['next_decision']="Still Alive"
            bets=bets._append(still_active_bets, ignore_index=True)
            bets = bets.query("security_id != 'Cash'")
            hit_ratio=len(bets[bets['performance']>0])/len(bets)
            win_loss_ratio=-bets[bets['performance']>0]['performance'].mean() /bets[bets['performance']<0]['performance'].mean()
            tab=[ptf.unique_id,len(bets),ptf.nav, ptf.performance,  hit_ratio,win_loss_ratio]
            behave=behave._append(pd.DataFrame([tab],columns=cols), ignore_index=True)

            winLossRatioData.append(win_loss_ratio)
            hitRatioData.append(hit_ratio)

            i += 1


        Metrics = ["Min", "Median", "Average", "Max"]

        total_return = formatPercentages(calculateMetrics(totalReturn, True))[4]
        volatility = formatPercentages(calculateMetrics(gg_perf['Volatility'], True))[4]
        IR = formatPercentages(calculateMetrics(gg_perf['IR'], True))[4]
        maxdrawdown = formatPercentages(calculateMetrics(gg_perf['MaxDrawdown'], True))[4]
        hit_ratio = calculateMetrics(hitRatiodata, False)[4]
        win_loss_ratio = calculateMetrics(winLossRatioData, False)[4]

        return total_return, volatility, maxdrawdown, IR, hit_ratio, win_loss_ratio 



        def _get_reward(self):

            # Defining the weights 
            w1 = 0.5  
            w2 = 0.2  
            w3 = 0.1  
            w4 = 0.1
            w5 = 0.05
            w6 = 0.05
            
            # Normalizing the values
            normalized_total_return = self.total_return / 100 
            normalized_volatility = self.volatility / 100
            normalized_max_drawdown = self.maxdrawdown / 100
            normalized_IR = self.IR / 100
            normalized_hit_ratio = self.hit_ratio / 100
            normalized_win_loss_ratio = self.win_loss_ratio / 100

            # Calculate the reward 
            reward = w1 * normalized_total_return + w2 * normalized_volatility + w3 * normalized_max_drawdown + w4 * normalized_IR + w5 * normalized_hit_ratio + w6 * normalized_win_loss_ratio
            
            return reward



