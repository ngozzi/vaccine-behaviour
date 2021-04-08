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
    cases  = pd.read_csv(path_to_data + country + "/epidemiological/cases.csv")
    deaths = pd.read_csv(path_to_data + country + "/epidemiological/deaths.csv")
    
    # import restriction    
    school_reductions = pd.read_csv(path_to_data + country + "/restrictions/school.csv")
    work_reductions   = pd.read_csv(path_to_data + country + "/restrictions/work.csv")
    oth_reductions    = pd.read_csv(path_to_data + country + "/restrictions/other_loc.csv")
    
    
    # create dict of data 
    country_dict = {"country"                 : country,
                    "contacts_work"           : work, 
                    "contacts_school"         : school,
                    "contacts_home"           : home,
                    "contacts_other_locations": other_locations,
                    "school_red"              : school_reductions,
                    "work_red"                : work_reductions,
                    "oth_red"                 : oth_reductions,
                    "Nk"                      : Nk,
                    "deaths"                  : deaths,
                    "cases"                   : cases}
    
    return country_dict



def get_beta(R0, mu, chi, omega, f, C, Nk):
    """
    This functions return beta for a SEIR model with age structure
        :param R0 (float): basic reproductive number
        :param mu (float): recovery rate
        :param chi (float): relative infectivity of P, A infectious
        :param omega (float): inverse of the prodromal phase
        :param f (float): probability of being asymptomatic
        :param C (matrix): contacts matrix
        :param Nk (array): n. of individuals in different age groups
        :return: returns the rate of infection beta
    """

    C_hat = np.zeros((C.shape[0], C.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C_hat[i, j] = (Nk[i] / Nk[j]) * C[i, j]

    max_eV = np.max([e.real for e in np.linalg.eig(C_hat)[0]])
    return R0 / (max_eV * (chi / omega + (1 - f) / mu + chi * f / mu))


    
    
def update_contacts(country_dict, date, baseline=False, scenario=0):
    
    # get baseline contacts matrices
    home      = country_dict["contacts_home"]
    work      = country_dict["contacts_work"]
    school    = country_dict["contacts_school"]
    oth_loc   = country_dict["contacts_other_locations"]
    
    # baseline contacts matrix
    if baseline==True: 
        return home + school + work + oth_loc
    
    # get year-week
    if date.isocalendar()[1] < 10:
        year_week = str(date.isocalendar()[0]) + "-0" + str(date.isocalendar()[1])
    else:
        year_week = str(date.isocalendar()[0]) + "-" + str(date.isocalendar()[1])
        
    # get work / other_loc reductions   
    work_reductions   = country_dict["work_red"]
    comm_reductions   = country_dict["oth_red"]
    school_reductions = country_dict["school_red"]
    school_reductions["date"] = pd.to_datetime(school_reductions["date"])
    
    if year_week <= "2021-11":
        omega_w = work_reductions.loc[work_reductions.year_week==year_week]["work_red"].values[0]
        omega_c = comm_reductions.loc[comm_reductions.year_week==year_week]["oth_red"].values[0]
        C1_school = school_reductions.loc[school_reductions.date==date]["C1_School closing"].values[0]

    else:
        if scenario == 0:
            omega_w   = work_reductions.loc[work_reductions.year_week=="2021-11"]["work_red"].values[0]
            omega_c   = comm_reductions.loc[comm_reductions.year_week=="2021-11"]["oth_red"].values[0]
            C1_school = school_reductions.loc[school_reductions.date==datetime(2021, 3, 21)]["C1_School closing"].values[0]
            
        elif scenario == 1:
            omega_w   = work_reductions.loc[work_reductions.year_week=="2021-11"]["work_red"].values[0] * 1.25
            omega_c   = comm_reductions.loc[comm_reductions.year_week=="2021-11"]["oth_red"].values[0]  * 1.25
            C1_school = school_reductions.loc[school_reductions.date==datetime(2021, 3, 21)]["C1_School closing"].values[0] - 1
            
        elif scenario == 2: 
            omega_w   = work_reductions.loc[work_reductions.year_week=="2021-11"]["work_red"].values[0] * 1.50
            omega_c   = comm_reductions.loc[comm_reductions.year_week=="2021-11"]["oth_red"].values[0]  * 1.50
            C1_school = school_reductions.loc[school_reductions.date==datetime(2021, 3, 21)]["C1_School closing"].values[0] - 2
            
        # check we are not going below zero
        if C1_school < 0: 
            C1_school = 0

    omega_s = (3 - C1_school) / 3
    
    # contacts matrix with reductions
    return home + (omega_s * school) + (omega_w * work) + (omega_c * oth_loc)
            
    

    
    
    