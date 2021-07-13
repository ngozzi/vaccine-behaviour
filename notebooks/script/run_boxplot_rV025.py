# libraries
import sys
sys.path.append("../models/")
from vaccine_behaviour import integrate_BV
from functions import import_country, get_totR, update_contacts
import pandas as pd 
import numpy as np 
import os 
from datetime import datetime, timedelta
import argparse
import pickle as pkl 


# epi params
eps   = 1.0 / 3.7
mu    = 1.0 / 2.5
omega = 1.0 / 1.5
chi   = 0.55
f     = 0.35 
IFR   = [0.00161 / 100, # 0-4  
         0.00161 / 100, # 5-9
         0.00695 / 100, # 10-14
         0.00695 / 100, # 15-19 
         0.0309  / 100, # 20-24 
         0.0309  / 100, # 25-29
         0.0844  / 100, # 30-34
         0.0844  / 100, # 35-39
         0.161   / 100, # 40-44 
         0.161   / 100, # 45-49 
         0.595   / 100, # 50-54 
         0.595   / 100, # 55-59 
         1.93    / 100, # 60-64
         1.93    / 100, # 65-69 
         4.28    / 100, # 70-74 
         6.04    / 100] # 75+

start_date = datetime(2021, 1, 1)
end_date   = datetime(2021, 6, 1)

scenario = 0 

def import_params(country, median):
    
    files  = os.listdir("../output/posterior/")
    files  = [file for file in files if country in file]
    if country == "Egypt":
        th = 0.4
    else:
        th = 0.3
    
    files = [file for file in files if "th" + str(th) in file]

    
    params_sampled = []
    for file in files:
        if params_sampled == []:
            params_sampled = np.load("../output/posterior/" + file, allow_pickle=True)["arr_0"]
        else: 
            params = np.load("../output/posterior/" + file, allow_pickle=True)["arr_0"]
            params_sampled = np.concatenate((params_sampled, params))

    R0_s    = np.array([p[0] for p in params_sampled]) 
    Delta_s = np.array([p[1] for p in params_sampled])
    ics_s   = np.array([p[4] for p in params_sampled]) 
    
    if median: 
        return np.median(R0_s), np.median(Delta_s), np.median(ics_s, axis=0), 
    return np.array(R0_s), np.array(Delta_s), np.array(ics_s)


# parse basin name
parser  = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('basin', type=str, help='name of the basin')
args    = parser.parse_args()
country = args.basin


alpha_s    = [0, 0.1, 1, 10, 100]
gamma      = 0.5
rs         = [1.3]
VE         = 0.9
VES        = 0.7
VEM        = 1 - (1 - VE) / (1 - VES)
rV         = 0.25 / 100
model      = "vaccine_rate"
iterations = 200


# initialize solutions
data          = dict()
data[country] = dict()
for vaccination_strategy in ["old_first", "homogeneous", "20-49_first"]:
    data[country][vaccination_strategy] = dict()
    for r in rs:
        data[country][vaccination_strategy][r] = dict()
        for alpha in alpha_s:
            data[country][vaccination_strategy][r][alpha] = []
            

# import country
country_dict = import_country(country, "../../data/countries/")

# import sampled params 
R0s, Deltas, icss = import_params(country, median=False)

# compute contacts matrices 
Cs = {}
date, dates = start_date, [start_date]
for i in range((end_date - start_date).days): 
    Cs[date] = update_contacts(country_dict, date, scenario=scenario)
    date += timedelta(days=1)
    dates.append(date)

    
for it in range(iterations):
    
    print(it)

    # sample params 
    rnd_run = np.random.randint(0, len(R0s))
    R0      = R0s[rnd_run]
    Delta   = Deltas[rnd_run]
    ics     = icss[rnd_run]
    
    ics_new = np.zeros((16, 20))
    ics_new[:, :12] = ics
    
    # run the baseline
    sol_baseline, Vt, ws = integrate_BV(ics_new, (end_date - start_date).days, R0, eps, mu, omega, chi, f, IFR, 
                                        Delta, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, Cs, country_dict["Nk"], "old_first", 
                                        model, dates)

    # iterate over vaccination strategy
    for vaccination_strategy in ["old_first", "homogeneous", "20-49_first"]:

        # iterate over r
        for r in rs:
            
            # iterate over alpha
            for alpha in alpha_s:
                
                # run solution
                solution, Vt, ws = integrate_BV(ics_new, (end_date - start_date).days, R0, eps, mu, omega, chi, f, IFR, 
                                                Delta, r, alpha, gamma, rV, VES, VEM, Cs, country_dict["Nk"], 
                                                vaccination_strategy, model, dates)
                delta_baseline = ((sol_baseline.sum(axis=0)[11, -1] - sol_baseline.sum(axis=0)[11, 0]) +
                                  (sol_baseline.sum(axis=0)[19, -1] - sol_baseline.sum(axis=0)[19, 0]))
                delta_solution = ((solution.sum(axis=0)[11, -1] - solution.sum(axis=0)[11, 0]) +
                                  (solution.sum(axis=0)[19, -1] - solution.sum(axis=0)[19, 0]))
                
                data[country][vaccination_strategy][r][alpha].append((delta_baseline - delta_solution) / delta_baseline)

# save
with open("../output/runs/boxplot_"  + country + "_VE" + str(VE) + "_rV" + str(rV) + "_scenario"+ str(scenario) + ".pkl", "wb") as file: 
    pkl.dump(data, file)                                                  