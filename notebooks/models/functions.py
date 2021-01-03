# libraries
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta



def get_totR(path, start_date, country):
    
    """
    This function import the total number or non-susceptible individuals for a given country up to start_date
        :param path (string): path to the data folder
        :param start_date (datetime): starting date
        :param country (string): country
        :return 
    """
    
    # import projections df
    df_inf = pd.read_csv(path + "/daily-new-estimated-infections-of-covid-19.csv")
    df_inf.Date = pd.to_datetime(df_inf.Date)
    
    # loc country and period
    df_inf_country = df_inf[(df_inf.Entity==country.replace("_", " ")) & (df_inf.Date<start_date)].reset_index(drop=True)

    cols = ['Daily new estimated infections of COVID-19 (ICL, mean)',
            'Daily new estimated infections of COVID-19 (IHME, mean)',
            'Daily new estimated infections of COVID-19 (YYG, mean)',
            'Daily new estimated infections of COVID-19 (LSHTM, median)']

    return df_inf_country[cols].sum().mean()



def load_epi_params(path):
    
    """
    This function import the epidemiological parameters
        :param path (string): path to the data folder
        :return IFR, mu, eps, Delta
    """
    
    # import json
    with open(path + "epi_params.json", "r") as fp:
        epi_params = json.load(fp)
        
    return epi_params["IFR"], epi_params["mu"], epi_params["eps"], epi_params["Delta"]
    


def import_country(country, path_to_data):
    
    """
    This function returns all data needed for a specific country
        :param country (string): name of the country 
        :param path_to_data (string): path to the countries folder
        :return dict of country data (country_name, work, school, home, other_locations, Nk, epi_data)
    """
    
    # import contacts matrix
    work            = np.loadtxt(path_to_data + country + "/contacts_matrix/contacts_work.csv", delimiter=",", skiprows=1)
    school          = np.loadtxt(path_to_data + country + "/contacts_matrix/contacts_school.csv", delimiter=",", skiprows=1)
    home            = np.loadtxt(path_to_data + country + "/contacts_matrix/contacts_home.csv", delimiter=",", skiprows=1)
    other_locations = np.loadtxt(path_to_data + country + "/contacts_matrix/contacts_other_locations.csv", delimiter=",", skiprows=1)

    # import demographic
    Nk = pd.read_csv(path_to_data + country + "/demographic/pop_5years.csv").total.values

    # import epidemiological data
    epi_data = pd.read_csv(path_to_data + country + "/epidemiological/ecdc_data.csv")
    epi_data["date"] = pd.to_datetime(epi_data["date"])
    epi_data.set_index("date", inplace=True)
    epi_data.sort_values(by="date", inplace=True)
    
    # create dict of data 
    country_dict = {"country"                 : country,
                    "contacts_work"           : work, 
                    "contacts_school"         : school,
                    "contacts_home"           : home,
                    "contacts_other_locations": other_locations,
                    "Nk"                      : Nk,
                    "epi_data"                : epi_data}
    
    return country_dict



def get_beta(R0, mu, C, Nk):
    
    """
    This functions return beta for a SEIR model with age structure
        :param R0 (float): basic reproductive number
        :param mu (float): recovery rate
        :param C (matrix): contacts matrix
        :param Nk (array): n. of individuals in different age groups
        :return: returns the rate of infection beta
    """
    
    C_hat = np.zeros((C.shape[0], C.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C_hat[i, j] = (Nk[i] / Nk[j]) * C[i, j]
    return R0 * mu / (np.max([e.real for e in np.linalg.eig(C_hat)[0]]))



def update_contacts(country_dict, date, w_r=None, oth_r=None, sch_r=None, scenario=None):
    
    """
    This function update the contacts according to restrictive measures given a country
        :param country_dict (dict): dictionary of country data
        :param date (datetime): current datetime 
        :param w_r (array, default=None): work reduction parameters (Sept., Oct., Nov.) 
        :param oth_r (array, default=None): other_locations reduction parameters (Sept., Oct., Nov.)
        :param sch_r (array, default=None): school reduction parameters (Sept., Oct., Nov.)
        :param scenario (int, default=None): reopening scenario (possible values: 1, 2, 3)
        :return: returns the contacts matrix for the country/date selected
    """
    
    # fit, projections, and scenario periods
    fit_start = datetime(2020, 7, 1)
    fit_end   = datetime(2020, 12, 1) # excluded
    
    proj_start = datetime(2020, 12, 1)
    proj_end   = datetime(2021, 1, 1) # excluded
    
    scenario_start = datetime(2021, 1, 1)
    scenario_end   = datetime(2021, 6, 1) # excluded
    
    # get baseline contacts matrices
    home            = country_dict["contacts_home"]
    work            = country_dict["contacts_work"]
    school          = country_dict["contacts_school"]
    other_locations = country_dict["contacts_other_locations"]
        
        
    # set reduction params 
    if date.month == 9:     # September
        alpha_w, alpha_oth, alpha_s = float(w_r[0]) / 100.0, float(oth_r[0]) / 100.0, float(sch_r[0]) / 100.0
        
    elif date.month == 10:  # October
        alpha_w, alpha_oth, alpha_s = float(w_r[1]) / 100.0, float(oth_r[1]) / 100.0, float(sch_r[1]) / 100.0
        
    elif date.month >= 11:  # November / December
        alpha_w, alpha_oth, alpha_s = float(w_r[2]) / 100.0, float(oth_r[2]) / 100.0, float(sch_r[2]) / 100.0
        
        
    # school summer and winter holidays
    if date >= datetime(2020, 8, 1) and date < datetime(2020, 9, 1):
        alpha_s = 0.0
    if date >= datetime(2020, 12, 21) and date < datetime(2021, 1, 7):
        alpha_s = 0.0
    
    
    # check in which period is given date  
    if date >= fit_start and date < fit_end:              ### FIT
        return home + alpha_w * work + alpha_oth * other_locations + alpha_s * school
    
    elif date >= proj_start and date < proj_end:          ### PROJECTIONS        
        return home + alpha_w * work + alpha_oth * other_locations + alpha_s * school
        
        
    elif date >= scenario_start and date < scenario_end:  ### SCENARIOS
    
        # strict
        if scenario == 1:
            
            # 25 / 25 / 50
            alpha_w   = float(25) / 100
            alpha_oth = float(25) / 100
            alpha_s   = float(50) / 100
            
            # winter holidays 
            if date >= datetime(2020, 12, 21) and date < datetime(2021, 1, 7):
                alpha_s = 0.0
                
            return home + alpha_w * work + alpha_oth * other_locations + alpha_s * school 
            
        # moderate 
        elif scenario == 2: 
                        
            # 25 / 25 / 50
            alpha_w   = float(50) / 100
            alpha_oth = float(50) / 100
            alpha_s   = float(75) / 100
            
            # winter holidays 
            if date >= datetime(2020, 12, 21) and date < datetime(2021, 1, 7):
                alpha_s = 0.0
                
            return home + alpha_w * work + alpha_oth * other_locations + alpha_s * school
        
        # mild 
        elif scenario == 3:
            
            # 75 / 75 / 100
            alpha_w   = float(75) / 100
            alpha_oth = float(75) / 100
            alpha_s   = float(100) / 100
            
            # winter holidays 
            if date >= datetime(2020, 12, 21) and date < datetime(2021, 1, 7):
                alpha_s = 0.0
                
            return home + alpha_w * work + alpha_oth * other_locations + alpha_s * school
     
    else: 
        return home + work + other_locations + school
            
    

    
    
    