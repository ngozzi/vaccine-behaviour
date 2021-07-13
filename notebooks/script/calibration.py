# libraries
import sys
sys.path.append("../models/")
from vaccine_behaviour import integrate_BV
from functions import import_country, get_totR, update_contacts
import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta
import argparse
import uuid


# datetimes
start_date = datetime(2020, 8, 31)
end_date = datetime(2021, 1, 4)


# error metric
def wmape(arr1, arr2):
    # weigthed mape
    return np.sum(np.abs(arr1 - arr2)) / np.sum(np.abs(arr1))


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


# behaviour params (turned off)
r = 1.0
alpha, gamma, rV, VES, VEM = 0.0, 0.0, 0.0, 0.0, 0.0


# number of compartment and age groups
ncomp = 20
nage  = 16


# parse basin name
#parser = argparse.ArgumentParser(description='Optional app description')
#parser.add_argument('basin', type=str, help='name of the basin')
#args = parser.parse_args()
#basin = args.basin
basin = "Italy"


# import country
country_dict = import_country(basin, "../../data/countries/")


# I0
new_pos = country_dict["cases"].loc[country_dict["cases"]["year_week"]=="2020-35"]["weekly_count"].values[0]


# R(t=0)
r0 = get_totR("../../data/", start_date, country_dict["country"]) / country_dict["Nk"].sum()
  
    
# pre-compute contacts matrices
Cs = {}
date, dates = start_date, [start_date]
for i in range((end_date - start_date).days): 
    Cs[date] = update_contacts(country_dict, date)
    date += timedelta(days=1)
    dates.append(date)


# add week of year
dates = [datetime(2020, 8, 31) + timedelta(days=d) for d in range(126)]
year_week = []
for date in dates: 
    y, w = date.isocalendar()[0], date.isocalendar()[1]
    if w < 9: 
        year_week.append(str(y) + "-0" + str(w))
    else: 
        year_week.append(str(y) + "-" + str(w))
        
        
# deaths real 
deaths_real = country_dict["deaths"].loc[(country_dict["deaths"]["year_week"]>="2020-36") & \
                                         (country_dict["deaths"]["year_week"]<="2020-53")]["weekly_count"].values

# simulate
params_sampled, solutions_sampled = [], []
for i in range(10000):
    
    if (i % 1000) == 0: 
        print(i)
    
    i0 = np.random.randint(int(0.5 * new_pos), int(3.5 * new_pos)) / country_dict["Nk"].sum()
    # S:0, SV:1, S_NC:2, SV_NC:3, E:4, P:5, A:6, I:7, RA:8, RI:9, D:10, DO: 11
    ics = np.zeros((nage, ncomp))
    # initialize
    for k in range(nage):
        ics[k, 4] = eps**(-1) / (mu**(-1) + omega**(-1) + eps**(-1)) * i0 * country_dict["Nk"][k]
        ics[k, 5] = omega**(-1) / (mu**(-1) + omega**(-1) + eps**(-1)) * i0 * country_dict["Nk"][k]
        ics[k, 6] = f * mu**(-1) / (mu**(-1) + omega**(-1) + eps**(-1)) * i0 * country_dict["Nk"][k]
        ics[k, 7] = (1 - f) * mu**(-1) / (mu**(-1) + omega**(-1) + eps**(-1)) * i0 * country_dict["Nk"][k]
        ics[k, 9] = r0 * country_dict["Nk"][k]
        ics[k, 0] = country_dict["Nk"][k] - ics[k, 4] - ics[k, 5] - ics[k, 6] - ics[k, 7] - ics[k, 9]

    R0 = np.random.uniform(0.8, 2.2)
    Delta = np.random.randint(14, 25)
    solution, Vt, vs = integrate_BV(ics, (end_date-start_date).days, R0, eps, mu, omega, chi, f, 
                                IFR, Delta, r, alpha, gamma, 
                                rV, VES, VEM, Cs, country_dict["Nk"], 
                                vaccination_strategy="homogeneous", model="vaccine_rate", 
                                dates=dates)
    
    # sum over age and create df
    solution_age = solution.sum(axis=0)
    df_deaths = pd.DataFrame(data={"deaths": np.diff(solution_age[11]), "dates": dates[1:]})
    df_deaths["year_week"] = year_week[1:]
    
    # resample weekly and compute wmape
    deaths_sim = df_deaths.groupby(by="year_week").sum().values.reshape(18)
    wmape_     = wmape(deaths_real, deaths_sim)
    if wmape_ < 0.3: 
        params_sampled.append([R0, Delta, i0, deaths_sim, solution[:,:,-1], wmape_])
        
        
# save
unique_filename = str(uuid.uuid4())
np.savez_compressed("../output/posterior/" + unique_filename + "_" + basin + ".npz", params_sampled)