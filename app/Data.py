
import os
import random
from datetime import datetime  
from datetime import timedelta
import copy
import pandas as pd
import datetime
import time



path=""
filename=path+ "SP500_us_blend1.xlsx"
filename=path+ "for_alfie.xlsx"


print("opening file", filename)
xl = pd.ExcelFile(filename)
sheets_name="prices" #xl.sheet_names[0]
all_prices=xl.parse(sheets_name)
all_prices['date']=all_prices['date'].dt.date
bmk="^GSPC"
bmk=all_prices.columns[1]
prices_bmk=all_prices[['date',bmk]].set_index('date')
prices_df=all_prices.drop(bmk, axis=1)
prices_df['Cash']=1.
prices_df=prices_df.ffill().set_index('date')
stock_universe=prices_df.columns

sheets_name="starting_holdings"
starting_holdings=xl.parse(sheets_name).set_index('security_id')['nominal']


sheets_name="dates"
simulation_dates_df=xl.parse(sheets_name)
simulation_dates=simulation_dates_df['date'].dt.date


ptf_id_ref = simulation_dates_df['ptf_id']
nav_ref = simulation_dates_df['nav']
performance_ref = simulation_dates_df['performance']
hit_ratio_ref = simulation_dates_df['hit_ratio']
win_loss_ratio_ref = simulation_dates_df['win_loss_ratio']
daily_performance_ref = simulation_dates_df['daily_performance']


xl.close()
start_date=simulation_dates.min()
end_date=simulation_dates.max()


list_available=prices_df.loc[start_date, prices_df.columns[:-1]].T.to_frame(name='price')
stock_universe=list_available.loc[list_available['price'].notnull()].index


bets_cols=["security_id","last_decision", "next_decision", "start_date",  "end_date", "performance"]
start_time= time.time()

portfolio_manager_behaviour={"style":
                                        {
                                        "buy":
                                            {
                                            "number_of_days":20,
                                            "momentum1":
                                                        {
                                                        "momentum_level_min":-0.05,
                                                        "momentum_level_max":0.025,
                                                        "percentage":0.5
                                                        },
                                            "momentum2":
                                                        {
                                                        "momentum_level_min":0.025,
                                                        "momentum_level_max":0.1,
                                                        "percentage":0.5
                                                        }
                                            },
                                        "sell":
                                            {
                                            "number_of_days":20,
                                            "momentum1":
                                                        {
                                                        "momentum_level_min":-0.05,
                                                        "momentum_level_max":0.025,
                                                        "percentage":0.5
                                                        },
                                            "momentum2":
                                                        {
                                                        "momentum_level_min":0.025,
                                                        "momentum_level_max":0.1,
                                                        "percentage":0.5
                                                        }
                                            },
                                        },
                                        
                    
                            "buy_behaviour":
                                            {"min_cash":0.01,
                                            "max_wght_buy":0.01,  # not a max 
                                            "nb_max_building_stock":1000,
                                            "buy_every_days":2},
                            "sell_behaviour":
                                            {"max_cash":0.5,
                                            "sell_every_days":1,
                                            },
                            "scale_up_behaviour":
                                            {"min_cash":0.01,
                                            "max_weight":0.01,
                                            "increment":0.025,
                                            "nb_max_building_stock":10,
                                            "scale_up_every_days":2,
                                            "weight_to_scale_up":0.0025
                                            },
        
                            "scale_down_behaviour":
                                            {"scale_down_every_days":2,
                                             "weight_to_scale_down":0.0025
                                            }
                            }