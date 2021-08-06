# libraries
import sys
sys.path.append("../models/")
from vaccine_behaviour import integrate_BV
from functions import import_country, get_totR, update_contacts
import pandas as pd 
import numpy as np 
import os 
from datetime import datetime, timedelta

# suppress warnings
import warnings
warnings.filterwarnings("ignore")


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
R0      = 1.15
Delta   = 17
i0      = 0.5 / 100.0
r0      = 10.0 / 100.0

# vaccine efficacy
VE         = 0.9
VES        = 0.7
VEM        = 1 - (1 - VE) / (1 - VES)

# dates
start_date = datetime(2021, 1, 1)
end_date   = datetime(2021, 12, 1)


# values of alpha and gamma
alpha_s = [10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3)]
gamma_s = [10**(-6), 10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1)]
data    = dict()

countries = ["Italy", "Canada", "Serbia", "Ukraine", "Egypt", "Peru"]
rs        = [1.3, 1.5]
rVs       = [1.0 / 100]
vaccination_strategy = "old_first"
models    = ["vaccine_rate", "constant_rate"]


for country in countries:
    
    print(country)
    data[country] = dict()
    
    # import country
    country_dict = import_country(country, "../../data/countries/")
    
    # initial conditions
    ics = np.zeros((16, 20))
    for k in range(16):
        ics[k, 4] = eps**(-1) / (mu**(-1) + omega**(-1) + eps**(-1)) * i0 * country_dict["Nk"][k]
        ics[k, 5] = omega**(-1) / (mu**(-1) + omega**(-1) + eps**(-1)) * i0 * country_dict["Nk"][k]
        ics[k, 6] = f * mu**(-1) / (mu**(-1) + omega**(-1) + eps**(-1)) * i0 * country_dict["Nk"][k]
        ics[k, 7] = (1 - f) * mu**(-1) / (mu**(-1) + omega**(-1) + eps**(-1)) * i0 * country_dict["Nk"][k]
        ics[k, 9] = r0 * country_dict["Nk"][k]
        ics[k, 0] = country_dict["Nk"][k] - ics[k, 4] - ics[k, 5] - ics[k, 6] - ics[k, 7] - ics[k, 9]
    
        
    # compute contacts matrices 
    Cs = {}
    C  = country_dict["contacts_home"]   + country_dict["contacts_work"] + \
         country_dict["contacts_school"] + country_dict["contacts_other_locations"]
    date, dates = start_date, [start_date]
    for i in range((end_date - start_date).days): 
        Cs[date] = C
        date += timedelta(days=1)
        dates.append(date)
        
    # run the baseline solution with no vaccine
    sol_baseline, Vt, ws = integrate_BV(ics, (end_date - start_date).days, R0, eps, mu, omega, chi, f, IFR, 
                                        Delta, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, Cs, country_dict["Nk"], "homogeneous",
                                        "vaccine_rate", dates)
        
    # run different rollout speeds
    for rV in rVs:
        print("\t", rV)
        data[country][rV] = dict()

        # run different r value
        for r in rs:
            print("\t\t", r)
            data[country][rV][r] = dict()
            
            # run different models
            for model in models:
                print("\t\t\t", model)
                data[country][rV][r][model] = np.zeros((len(alpha_s), len(gamma_s)))
                
                # run different alpha and gammaa
                for i in range(len(alpha_s)): 
                    print("\t\t\t\t", alpha_s[i])
                    for j in range(len(gamma_s)): 
                        solution, Vt, ws = integrate_BV(ics, (end_date - start_date).days, R0, eps, mu, omega, chi, f, IFR,
                                                        Delta, r, alpha_s[i], gamma_s[j], rV, VES, VEM, Cs, country_dict["Nk"],
                                                        vaccination_strategy, model, dates)
                        
                        
                        delta_baseline = (sol_baseline.sum(axis=0)[11, -1] - sol_baseline.sum(axis=0)[11, 0])
                        delta_solution = (solution.sum(axis=0)[11, -1] - solution.sum(axis=0)[11, 0])
                        data[country][rV][r][model][i,j] = (delta_baseline - delta_solution) / delta_baseline

import pickle as pkl
with open("../output/phase_space.pkl", "wb") as file: 
    pkl.dump(data, file)                                                   
