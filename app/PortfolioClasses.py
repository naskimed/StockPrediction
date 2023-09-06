import os
from mesa.space import MultiGrid
from mesa import Agent, Model
from mesa.time import RandomActivation,BaseScheduler
import random
from mesa.datacollection import DataCollector
from datetime import datetime  
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import pandas as pd 
import seaborn as sns
import time
import copy


class Portfolio(Model):
    """ the portfolio is like a madnate where we specify 
        -the period
        - the universe from whcih a manager can select stocks
        - teh number of portfolio manager we want to delegate to 
        """
    def __init__(self, number_of_managers_per_portfolio, starting_holdings, portfolio_manager_behaviour, simulation_dates, prices_df, prices_bmk):
        self.schedule =BaseScheduler(self)
        self.running = True
        self.date_ptf= simulation_dates[0]
        self.date_ptf_t_plus= simulation_dates[1]
        self.useful_prices=prices_df.loc[self.date_ptf].T.to_frame(name='prices').dropna()
        self.useful_prices_next=prices_df.loc[self.date_ptf_t_plus].T.to_frame(name='prices').dropna()
        self.prices = prices_df
        self.pricesBMK = prices_bmk
        self.simulationDates = simulation_dates
        for portfolio_id in range(0,number_of_managers_per_portfolio):
                ptf=PortfolioManager(portfolio_id,self, starting_holdings, portfolio_manager_behaviour, simulation_dates, prices_df, prices_bmk)
                self.schedule.add(ptf)



        
        self.datacollector = DataCollector(
            #model_reporters={"Gini": compute_gini},  # A function to call
            agent_reporters={"date": lambda a: getattr(a.model, 'date_ptf', None),
                            "NAV": lambda a: getattr(a, 'nav', None),
                            "holdings":lambda a: getattr(a, 'holdings', None).copy(),
                            "holdings_weights": lambda a: getattr(a, 'holdings_weights', None),

                            "performance": lambda a: getattr(a, 'performance', None),
                            "performance_attribution_active_bets": lambda a: getattr(a, 'performance_attribution_active_bets', None),
                            "useful_prices":lambda a: getattr(a.model, 'useful_prices', None),
                            "bets":lambda a: getattr(a, 'bets', None),
                            "buy_pipeline":lambda a: getattr(a, 'buy_pipeline', None),
                            "sell_pipeline":lambda a: getattr(a, 'sell_pipeline', None),

                            }  # An agent attribute
                            )
        
    def getStockUniverse(self):
        ## drawdown  paramters
        stop_loss=-100000#0.005
        current_date=self.date_ptf
        
        
        self.useful_prices=self.prices.loc[current_date].T.to_frame(name='prices').dropna()
        next_date=self.date_ptf_t_plus
        self.useful_prices_next=self.prices.loc[next_date].T.to_frame(name='prices').dropna()

        #self.buy_pipelin=portfolio.useful_prices.loc[portfolio.useful_prices['prices'].notnull()].index.to_list()
        
        #self.sell_pipeline=[]# [obj for obj in portfolio.holdings.keys() if (portfolio.performance_dd.get(obj, 0) <stop_loss) ]
        

    def step(self):
        print("day ", self.schedule.steps)
        self.schedule.step()
        self.date_ptf=self.simulationDates[1 + np.where(self.simulationDates>=self.date_ptf)[0][0]]  
        self.date_ptf_t_plus=self.simulationDates[1 + np.where(self.simulationDates>=self.date_ptf)[0][0]]  

        self.getStockUniverse()

        self.datacollector.collect(self)
        
        

  
class PortfolioManager(Agent):
    """ An agent with fixed initial wealth."""
    def __init__(self, unique_id, model,  starting_holdings, behaviours, simulation_dates, prices_df, prices_bmk):
        super().__init__(unique_id, model)
        self.style = behaviours['style']
        #print(self.style)
        #print(self.style['type'])
        #print(self.style['parameters']['momentum_level_min'])
        #self.date = simulation_dates[0]
        #self.useful_prices=prices_df.loc[self.date].T.to_frame(name='prices')
        self.holdings=starting_holdings.copy()
        self.strike_nav_calculate_weights()
        self.calculate_weights()
        self.performance=0.0
        #self.controlling_pm_id=controlling_pm
        self.position_building_stock=[]
        self.buy_behaviour=behaviours['buy_behaviour']
        self.sell_behaviour=behaviours['sell_behaviour']
        self.scale_up_behaviour=behaviours['scale_up_behaviour']
        self.scale_down_behaviour=behaviours['scale_down_behaviour']
        
        zero_perf=d = pd.Series(0, index=starting_holdings.index, name='performance')
        self.simulationDates = simulation_dates
        self.prices = prices_df
        self.pricesBMK = prices_bmk
     
              
        self.performance_attribution=zero_perf.copy()
        self.performance_attribution_active_bets=zero_perf.copy()

        #self.performance_over_dd_period=dict(zip(self.holdings.keys(), [[] for k in self.holdings.keys()]))
        #self.performance_dd=dict(zip(self.holdings.keys(), [0.0 for k in self.holdings.keys()]))
        #self.nb_stop_loss=0
        bets0=pd.DataFrame(self.model.date_ptf , index=starting_holdings.index, columns=["start_date"])
        bets0['last_decision']="Legacy"
        bets0['performance']=0
        self.bets={"active":bets0,
                   "closed": pd.DataFrame(None, columns=["security_id","last_decision", "next_decision", "start_date",  "end_date", "performance"]) 
                   }
        self.buy_pipeline = []
        self.sell_pipeline = []
    
    def getStocksPipelines(self,pipeline_size):
        stop_loss=0#0.005
        current_date=self.model.date_ptf
        #print(self.style['type']=='momentum')
        if self.style['buy']!= None:  
            momentum1_level_min = self.style['buy']['momentum1']['momentum_level_min']
            momentum2_level_min = self.style['buy']['momentum2']['momentum_level_min']
            momentum1_level_max = self.style['buy']['momentum1']['momentum_level_max']
            momentum2_level_max = self.style['buy']['momentum2']['momentum_level_max']
            number_of_days=self.style['buy']['number_of_days']
            where_are_we=np.where(self.simulationDates>current_date)[0][0]
            #print(momentu_level_min)
            #print(momentu_level_max)

            cols = set(self.model.useful_prices.index.values) - set(self.holdings.index.values)
            cols = list(cols)
            if where_are_we>number_of_days:
                prev= self.simulationDates[-number_of_days + np.where(self.simulationDates>current_date)[0][0]]
                prices_3_m_before=self.prices.loc[prev, cols].T.to_frame(name='prices').dropna()
                diff=(self.model.useful_prices/prices_3_m_before) -(self.pricesBMK.loc[current_date].values[0]/self.pricesBMK.loc[prev].values[0])
                #print((self.model.useful_prices/prices_3_m_before) )
                #print(prices_bmk.loc[current_date, bmk]/prices_bmk.loc[prev, bmk])
                momentum1=diff[(diff['prices']>momentum1_level_min) & (diff['prices']<momentum1_level_max)]
                momentum2=diff[(diff['prices']>momentum2_level_min) & (diff['prices']<momentum2_level_max)]
                #print(diff.shape, momentum.shape)
                #print( momentum)

                size1 = int(round(pipeline_size * self.style['buy']['momentum1']['percentage'], 0))
                size2 = pipeline_size - size1
                if len(momentum1) or len(momentum2)>0:
                    momentum1_items = list(momentum1.index)
                    momentum2_items = list(momentum2.index)

                    buy_pipeline_1 = random.sample(momentum1_items, min(size1, len(momentum1_items)))
                    # Unique items for momentum2
                    remaining_items = list(set(momentum2_items) - set(buy_pipeline_1))
                    buy_pipeline_2 = random.sample(remaining_items, min(size2,len(remaining_items)))
                    self.buy_pipeline = buy_pipeline_1 + buy_pipeline_2
                    #print(len(self.buy_pipeline))
                else:
                    self.buy_pipeline=[]
            else:
                self.buy_pipeline=[]
        else:
            self.buy_pipeline=random.choices(list(self.model.useful_prices.index), k=pipeline_size)
        

        if self.style['sell'] != None:
            momentum1_level_min = self.style['sell']['momentum1']['momentum_level_min']
            momentum2_level_min = self.style['sell']['momentum2']['momentum_level_min']
            momentum1_level_max = self.style['sell']['momentum1']['momentum_level_max']
            momentum2_level_max = self.style['sell']['momentum2']['momentum_level_max']
            number_of_days=self.style['sell']['number_of_days']
            where_are_we=np.where(self.simulationDates>current_date)[0][0]

            if where_are_we>number_of_days:
                prev= self.simulationDates[-number_of_days + np.where(self.simulationDates>current_date)[0][0]]
                prices_3_m_before=self.prices.loc[prev, self.holdings.keys()].T.to_frame(name='prices').dropna()
                diff=(self.model.useful_prices.loc[self.holdings.keys()]/prices_3_m_before) -(self.pricesBMK.loc[current_date].values[0]/self.pricesBMK.loc[prev].values[0])
                #print(f'self.model.useful_prices.loc[self.holdings.keys()]: {self.model.useful_prices.loc[self.holdings.keys()]}')
                #print(f'prices_3_m_before: {prices_3_m_before}')
                
                momentum1=diff[(diff['prices']>momentum1_level_min) & (diff['prices']<momentum1_level_max)]
                momentum2=diff[(diff['prices']>momentum2_level_min) & (diff['prices']<momentum2_level_max)]

                size1 = int(round(len(self.holdings.keys()) * self.style['sell']['momentum1']['percentage'], 0))
                size2 = len(self.holdings.keys()) - size1
                if len(momentum1) or len(momentum2)>0:
                    momentum1_items = list(momentum1.index)
                    momentum2_items = list(momentum2.index)

                    sell_pipeline_1 = random.sample(momentum1_items, min(size1, len(momentum1_items)))
                    #print(f'Sell pipeline1: {sell_pipeline_1}')
                    remaining_items = list(set(momentum2_items) - set(sell_pipeline_1))
                    sell_pipeline_2 = random.sample(remaining_items, min(size2,len(remaining_items)))
                    #print(f'Sell pipeline2: {sell_pipeline_2}')
                    self.sell_pipeline = sell_pipeline_1 + sell_pipeline_2
                    #print(f'Sell pipeline: {self.sell_pipeline}')
                    #print(f'Self holdings keys: {self.holdings.keys()}')
                    
                else:
                    self.sell_pipeline=[]
                    #print("!(len(momentum1) or len(momentum2)>0)")
            else:
                self.sell_pipeline=[]
                #print("where_are_we<=number_of_days")
        else:
            self.sell_pipeline=[obj for obj in self.holdings.keys() ]
            #print("self.style['sell'] == None")


        '''
        stop_loss=0#0.005
        current_date=self.model.date_ptf
        if self.style['type']=='momentum':   
            momentu_level_min=self.style['parameters']['momentum_level_min']
            momentu_level_max=self.style['parameters']['momentum_level_max']
            number_of_days=self.style['parameters']['number_of_days']
            where_are_we=np.where(simulation_dates>current_date)[0][0]
            if where_are_we>number_of_days:
                prev= simulation_dates[-number_of_days + np.where(simulation_dates>current_date)[0][0]]
                prices_3_m_before=prices_df.loc[prev].T.to_frame(name='prices').dropna()
                diff=(self.model.useful_prices/prices_3_m_before) -(prices_bmk.loc[current_date, bmk]/prices_bmk.loc[prev, bmk])
                #print((self.model.useful_prices/prices_3_m_before) )
                #print(prices_bmk.loc[current_date, bmk]/prices_bmk.loc[prev, bmk])
                momentum=diff[(diff['prices']>momentu_level_min) & (diff['prices']<momentu_level_max)]
                print(diff.shape, momentum.shape)
                #print( momentum)
                
                if len(momentum)>0:
                    self.buy_pipeline=random.choices(list(momentum.index), k=pipeline_size)
                else:
                    self.buy_pipeline=[]
            else:
                self.buy_pipeline=[]
        else:
            self.buy_pipeline=random.choices(list(self.model.useful_prices.index), k=pipeline_size)
        
        self.sell_pipeline=[]# [obj for obj in portfolio.holdings.keys() if (portfolio.performance_dd.get(obj, 0) <stop_loss) ]
        self.sell_pipeline= [obj for obj in self.model.useful_prices.index if (self.performance_attribution_active_bets.get(obj, 0) <stop_loss) ]
        '''

        
        
    def buy_stock(self):
        
        min_cash=self.buy_behaviour['min_cash']
        max_wght_buy=self.buy_behaviour['max_wght_buy']
        nb_max_building_stock=self.buy_behaviour['nb_max_building_stock']
        buy_every_days=self.buy_behaviour['buy_every_days']
        buy_pipeline1=list(set(self.buy_pipeline) -set(self.holdings.index))
        
        if ((self.holdings_weights['Cash']>min_cash) and (len(self.position_building_stock) <nb_max_building_stock) and (len(buy_pipeline1)>0) and (self.model.schedule.steps >1)): 
            
            if (self.model.schedule.steps %buy_every_days ==0) :
                
                stock_to_buy= random.choice(list(buy_pipeline1))
                
                price=self.model.useful_prices.loc[stock_to_buy, 'prices']  
                wght_to_buy=min(self.holdings_weights['Cash']- min_cash, max_wght_buy)
                
                
                self.position_building_stock.append(stock_to_buy)  
                self.holdings_weights[stock_to_buy]=wght_to_buy 
                self.holdings[stock_to_buy]=wght_to_buy *self.nav/price 
                self.performance_attribution_active_bets[stock_to_buy]=0
                       
                self.holdings_weights['Cash']=self.holdings_weights['Cash'] -wght_to_buy
                self.holdings['Cash']=self.holdings_weights['Cash'] *self.nav
                
                self.bets['active'].loc[stock_to_buy]= [self.model.date_ptf, "Buy", 0]

                
                
    def scale_up_stock(self):
        min_cash=self.scale_up_behaviour['min_cash']
        # building positions
        max_weight=self.scale_up_behaviour['max_weight']
        increment=self.scale_up_behaviour['increment']
        nb_max_building_stock=self.scale_up_behaviour['nb_max_building_stock']
        scale_up_every_days=self.scale_up_behaviour['scale_up_every_days']
        weight_to_scale_up=self.scale_up_behaviour['weight_to_scale_up']
        # building positions
        
        for sec in self.position_building_stock:
                #print("Sec",sec," Cash ",  portfolio.holdings_weights['Cash'], " Sec", portfolio.holdings_weights[sec] )
                if (self.holdings_weights[sec]<max_weight):
                    if (self.holdings_weights['Cash']>min_cash):
                        #if (portfolio.holdings_weights[sec]<max_weight):
                        stock_to_buy= sec
                        #price=prices_df[prices_df['date']==portfolio.date][stock_to_buy].values[0]


                        price=self.model.useful_prices.loc[stock_to_buy, 'prices']  
                        wght_to_buy=min(self.holdings_weights['Cash']- min_cash, increment)
                        #print("in buys", stock_to_buy, price, wght_to_buy)
                        
                        old_wght=self.holdings_weights[stock_to_buy]
                        
                        self.holdings_weights[stock_to_buy]=wght_to_buy+old_wght

                        self.holdings[stock_to_buy]=self.holdings_weights[stock_to_buy] *self.nav/price
                        
                        self.holdings_weights['Cash']=self.holdings_weights['Cash'] -wght_to_buy
                        self.holdings['Cash']=self.holdings_weights['Cash'] *self.nav
                        ##print("on ptf ", self.unique_id,"building", sec, "bought ", wght_to_buy, " reached ", self.holdings_weights[stock_to_buy])
                        #print("#################  #####", portfolio.holdings_weights)

                    else:
                        #print("not enough cash to built up position")
                        a="may be at a alater stage force sell to buy"
                else: 
                        self.position_building_stock=list(set(self.position_building_stock) - set([sec]))
                        #print("building position finished", sec,  "new weight",portfolio.holdings_weights[sec], "new list to build", portfolio.position_building_stock, "for ptf" , portfolio.unique_id)
        
        # regular scale up if cash
        if ((self.holdings_weights['Cash']>min_cash) and (self.model.schedule.steps >1)):
            #buy_pipeline1=list(set(self.buy_pipeline) -set(self.holdings.index))
            #print("size pipeline", len(portfolio.buy_pipeline))
            #scale_up_pipeline=list(buy_pipeline)-portfolio.position_building_stock
            if (self.model.schedule.steps %scale_up_every_days ==0) :
                stock_to_scale_up= random.choice(list(set(self.holdings.index)-set(['Cash'])))
                
                
                #print("check", list(set(self.holdings.index)-set(['Cash'])))

                #price=prices_df[prices_df['date']==portfolio.date][stock_to_buy].values[0]
                
                
                price=self.model.useful_prices.loc[stock_to_scale_up, 'prices']  
                wght_to_scale_up=min(self.holdings_weights['Cash']- min_cash, weight_to_scale_up)
                #print("in buys", stock_to_buy, price, wght_to_buy)
                
                old_wght=self.holdings_weights[stock_to_scale_up]
                
                self.holdings_weights[stock_to_scale_up]=wght_to_scale_up+old_wght

                self.holdings[stock_to_scale_up]=self.holdings_weights[stock_to_scale_up] *self.nav/price
                
                self.holdings_weights['Cash']=self.holdings_weights['Cash'] -wght_to_scale_up
                self.holdings['Cash']=self.holdings_weights['Cash'] *self.nav
                new_closed=pd.DataFrame({"security_id":stock_to_scale_up,
                                        "last_decision":self.bets['active'].loc[stock_to_scale_up, "last_decision"],
                                        "next_decision":"Scale Up",
                                        "start_date": self.bets['active'].loc[stock_to_scale_up, "start_date"],
                                        "end_date": self.model.date_ptf, 
                                        "performance":self.performance_attribution_active_bets[stock_to_scale_up]
                                         }, index=[0])
                self.bets['closed']=self.bets['closed']._append(new_closed, ignore_index=True)   ### need tp change to reset teh performance periodically
                #self.bets['active'][stock_to_scale_up]= self.model.date_ptf
                #print(f"Bets active: {self.bets['active']}")
                #print(f"Bets closed: {self.bets['closed']}")
                #print(f"Stock to scale up: {stock_to_scale_up}")
               # print(f"self.bets['active'].loc[stock_to_scale_up]:{self.bets['active'].loc[stock_to_scale_up]}")
                self.bets['active'].loc[stock_to_scale_up]= [self.model.date_ptf, "Scale Up", 0]
                self.performance_attribution_active_bets[stock_to_scale_up]=0 

    def scale_down_stock(self):
        ### sclae dow radomly randomly
        scale_down_every_days=self.scale_down_behaviour['scale_down_every_days']
        weight_to_scale_down=self.scale_down_behaviour['weight_to_scale_down']
        #stock_to_sell=None
        scale_d_list=list(set(self.holdings.keys())-set(['Cash'])- set(self.position_building_stock))

        if ((len(scale_d_list)>0)  and (self.model.schedule.steps % scale_down_every_days==0) and (self.model.schedule.steps >1)) :
            #scale_d_list=list(set(self.holdings.keys())-set(['Cash'])- set(self.position_building_stock))
            stock_to_scale_d= random.choice(scale_d_list)
            #print("from", self.unique_id, 'to sell ', stock_to_sell, "used list", sell_list, "in list",self.holdings.index,"#Holding Weight#",self.holdings_weights , "building", self.position_building_stock )

            price=self.model.useful_prices.loc[stock_to_scale_d, 'prices']  
            #wght_to_scale_d=min(self.holdings_weights['Cash']- min_cash, increment)
            
            old_wght=self.holdings_weights[stock_to_scale_d]
            ##print(" Cadaite to scle d", stock_to_scale_d, old_wght, weight_to_scale_down)
            
            if (old_wght > weight_to_scale_down): # still weight left after scale d
                self.holdings_weights[stock_to_scale_d]=old_wght -weight_to_scale_down

                self.holdings[stock_to_scale_d]=self.holdings_weights[stock_to_scale_d] *self.nav/price
                
                self.holdings_weights['Cash']=self.holdings_weights['Cash'] +weight_to_scale_down
                self.holdings['Cash']=self.holdings_weights['Cash'] *self.nav
                new_closed=pd.DataFrame({"security_id":stock_to_scale_d,
                                        "last_decision":self.bets['active'].loc[stock_to_scale_d, "last_decision"],
                                        "next_decision":"Scale Down",
                                        "start_date": self.bets['active'].loc[stock_to_scale_d, "start_date"],
                                        "end_date": self.model.date_ptf, 
                                        "performance":self.performance_attribution_active_bets[stock_to_scale_d]
                                         }, index=[0])
                self.bets['closed']=self.bets['closed']._append(new_closed, ignore_index=True)   ### need tp change to reset teh performance periodically
                self.bets['active'].loc[stock_to_scale_d]= [self.model.date_ptf, "scale Down",0]
                self.performance_attribution_active_bets[stock_to_scale_d]=0

                ##print("on ptf ", self.unique_id, "Scale dow", stock_to_scale_d, "New Cash ",self.holdings_weights['Cash'],  "old weight" , old_wght, "mew Weight ", self.holdings_weights[stock_to_scale_d])#closed",len( self.bets['closed']))

            
            else: # eed to sell alll 
                self.holdings_weights.loc['Cash']=self.holdings_weights.loc['Cash']+old_wght
                self.holdings.loc['Cash']=self.holdings_weights.loc['Cash'] *self.nav
                #print("sold", stock_to_scale_d, "New Cash ",self.holdings_weights['Cash'])
                
                new_closed=pd.DataFrame({"security_id":stock_to_scale_d,
                                        "last_decision":self.bets['active'].loc[stock_to_scale_d, "last_decision"],
                                        "next_decision":"Sell",
                                         "start_date": self.bets['active'].loc[stock_to_scale_d, "start_date"],
                                         "end_date": self.model.date_ptf, 
                                         "performance":self.performance_attribution_active_bets[stock_to_scale_d]
                                         }, index=[0])
                self.bets['closed']=self.bets['closed']._append(new_closed, ignore_index=True)   ### need tp change to reset teh performance periodically
                
                self.holdings_weights.drop(stock_to_scale_d, axis=0, inplace=True)
                self.holdings.drop(stock_to_scale_d, axis=0, inplace=True)
                self.bets['active'].drop(stock_to_scale_d, axis=0,  inplace=True)
                self.performance_attribution_active_bets.drop(stock_to_scale_d, axis=0,  inplace=True)
  
    
    def sell_stock(self):
        ### sell randomly
        max_cash=self.sell_behaviour['max_cash']
        sell_every_days=self.sell_behaviour['sell_every_days']
        stock_to_sell=None
        #sell_list=list(set(self.holdings.keys())-set(['Cash'])- set(self.position_building_stock))
        sell_list=list(set(self.sell_pipeline)-set(['Cash'])- set(self.position_building_stock))

        #print(self.holdings_weights)
        #print("sell", (self.holdings_weights['Cash']>max_cash) , (len(sell_list)>0)  ,(self.model.schedule.steps % sell_every_days==0) , (self.model.schedule.steps >1))
        if ((self.holdings_weights['Cash']<max_cash) and (len(sell_list)>0)  and (self.model.schedule.steps % sell_every_days==0) and (self.model.schedule.steps >1)) :
            #sell_list=list(set(self.holdings.keys())-set(['Cash'])- set(self.position_building_stock))
            stock_to_sell= random.choice(sell_list)
            #print("from", self.unique_id, 'to sell ', stock_to_sell, "used list", sell_list, "in list",self.holdings.index,"#Holding Weight#",self.holdings_weights , "building", self.position_building_stock )
            
            wght_to_sell=self.holdings_weights.loc[stock_to_sell]
                      
            
            self.holdings_weights.loc['Cash']=self.holdings_weights.loc['Cash']+wght_to_sell
            self.holdings.loc['Cash']=self.holdings_weights.loc['Cash'] *self.nav
            #print("sold", stock_to_sell, "New Cash ",self.holdings_weights['Cash'])
            
            new_closed=pd.DataFrame({"security_id":stock_to_sell,
                                    "last_decision":self.bets['active'].loc[stock_to_sell, "last_decision"],
                                    "next_decision":"Sell",
                                    "start_date": self.bets['active'].loc[stock_to_sell, "start_date"],
                                    "end_date": self.model.date_ptf, 
                                    "performance":self.performance_attribution_active_bets[stock_to_sell]
                                             }, index=[0])
            self.bets['closed']=self.bets['closed']._append(new_closed, ignore_index=True)   ### need tp change to reset teh performance periodically
            
            #print("sold", stock_to_sell, "New Cash ",self.holdings_weights['Cash'], " # closed",len( self.bets['closed']))
            self.holdings_weights.drop(stock_to_sell, axis=0, inplace=True)
            self.holdings.drop(stock_to_sell, axis=0, inplace=True)
            self.bets['active'].drop(stock_to_sell, axis=0,  inplace=True)
            self.performance_attribution_active_bets.drop(stock_to_sell, axis=0,  inplace=True)

            
            #print("on ptf ", self.unique_id, "holdings", self.holdings,  "Sell ", stock_to_sell, "New Cash ",self.holdings_weights['Cash'], " # closed",len( self.bets['closed']))

    
    def strike_nav_calculate_weights(self):
        
        prices_df_local=self.model.useful_prices
        #prices_of_holdings=prices_df_local.loc[self.holdings]['prices']
        #prices_df_local['value']=prices_df_local['holdings']*prices_df_local['prices']
        ## the next assume we ahve all prices
        self.nav=  sum (self.holdings *prices_df_local.loc[self.holdings.index]['prices'])
        self.holdings_weights=(self.holdings *prices_df_local.loc[self.holdings.index]['prices'])/self.nav
    
    def calculate_weights(self):
        #prices_df_local=prices_df[prices_df['date']==self.date]
        prices_df_local=self.model.useful_prices
        
        prices_df_local['holdings']=pd.Series(self.holdings)#[self.holdings[sec]  for sec in self.holdings.keys()]
        prices_df_local['value']=prices_df_local['holdings']*prices_df_local['prices']/self.nav
        self.holdings_weights=prices_df_local[['value']].to_dict()['value']

    
    def calculate_perf(self):
        dd_period=20
        datet=self.model.date_ptf
        t_plus_one=self.model.date_ptf_t_plus
        prices_df_local=self.prices.loc[[datet,t_plus_one], self.holdings.keys()].T
        perf_bmk=self.pricesBMK.loc[t_plus_one].values[0]/self.pricesBMK.loc[datet].values[0]
        pt=self.model.useful_prices.loc[self.holdings.index]['prices'] 
        ptplus=self.model.useful_prices_next.loc[self.holdings.index]['prices']
        prices_df_local['performance']=(((ptplus/pt- perf_bmk)))*self.holdings_weights
        prices_df_local['previous_performance']=self.performance_attribution_active_bets
        prices_df_local['new_performance']=prices_df_local['performance']+prices_df_local['previous_performance']
        self.bets['active']['performance']=prices_df_local.loc[self.bets['active'].index,'new_performance']
        self.performance_attribution_active_bets=prices_df_local['new_performance']
        mask=prices_df_local.index!='Cash'
        self.performance= self.performance+prices_df_local.loc[mask,'performance'].sum()
   
        
    def receive_inflows(self, inflow):
        self.holdings["Cash"]=self.holdings["Cash"]+inflow


    def pay_outflow(self, outflow):
         self.holdings["Cash"]=self.holdings["Cash"]-outflow


    def step(self):
        self.getStocksPipelines(400)
       #print(" I am a PM", self.unique_id, "Buy Pipeline ",  self.buy_pipeline)

        #print(" USEFUL PR", self.model.useful_prices)        
        self.sell_stock()
        self.scale_up_stock()
        self.scale_down_stock()
        self.buy_stock()


        #inflow=self.nav*random.uniform(0,0.005)
        #self.receive_inflows(inflow)  
        self.strike_nav_calculate_weights()
        #print(" I am a PM", self.unique_id, "nav ",  self.nav, "weights ", self.holdings_weights)
        self.calculate_perf()       
        #print(" I am a portfolio", self.unique_id, "nav %.2f " %self.nav, "Cash %.3f" %self.holdings_weights['Cash']," Perf  %.4f" %self.performance, "# holdig ", len(self.holdings_weights), "closed ",  list(self.bets['closed'].groupby('next_decision').count().index),list(self.bets['closed'].groupby('next_decision').count()['security_id']))
        #print(" I am a portfolio", self.unique_id,self.holdings)
        #print("day ", self.model.schedule.steps, "new active on ",self.unique_id, len(self.bets['active'])) # , self.bets['active'])

