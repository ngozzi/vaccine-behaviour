# libraries
from scipy.integrate import odeint
import numpy as np
from datetime import datetime, timedelta
from functions import update_contacts, get_beta


# n. of compartments
ncomp = 6
 

def SEIRD_system(y, t, beta, eps, mu, Delta, C, Nk, IFR):
    
    """
    This function defines the system of differential equations
        :param y (array): compartment values at time t
        :param t (float): current time step
        :param beta (float): infection rate
        :param eps (float): incubation rate
        :param mu (float): recovery rate
        :param Delta (float): inverse of the rate of D -> DO
        :param C (matrix): contacts matrix
        :param Nk (array): number of individuals in different age groups
        :param IFR (array): age-stratified infection fatality rate
        :return: returns the system dydt of differential equations

    Note:
        - age groups must be ordered sequentially (from youngest to oldest)
        - for each age group compartments must be ordered like this: S, E, I, R, D, DO
    """

    # number of age groups
    nage = len(C)

    # system
    dydt = []

    # iterate over age groups and define for each one the respective differential equations
    for age1 in range(nage):

        # first we compute all interaction terms with other age groups
        S_to_E = 0
        for age2 in range(nage):
            S_to_E += C[age1][age2] * y[(ncomp * age2) + 2] / Nk[age2]   # \sum_k' C_kk' I_k' / N_k'

        ### EQUATION FOR S ###
        dydt.append(-beta * S_to_E * y[(ncomp * age1) + 0])         # S -> E

        ### EQUATION FOR E ###
        dydt.append(+beta * S_to_E * y[(ncomp * age1) + 0]          # S -> E
                    -eps * y[(ncomp * age1) + 1])                   # E -> I

        ### EQUATION FOR I ###
        dydt.append(+eps * y[(ncomp * age1) + 1]                    # E -> I
                    -mu * (1 - IFR[age1]) * y[(ncomp * age1) + 2]   # I -> R
                    -mu * IFR[age1] * y[(ncomp * age1) + 2] )       # I -> D

        ### EQUATION FOR R ###
        dydt.append(+mu * (1 - IFR[age1]) * y[(ncomp * age1) + 2])  # I -> R
        
        ### EQUATION FOR D ### 
        dydt.append(+mu * IFR[age1] * y[(ncomp * age1) + 2]         # I -> D
                    -(1 / Delta) * y[(ncomp * age1) + 4])           # D -> DO
        
        ### EQUATION FOR DO ###
        dydt.append(+(1 / Delta) * y[(ncomp * age1) + 4])           # D -> DO
                      
    return dydt



def integrate_SEIRD(e0, i0, r0, initial_date, T, R0, eps, mu, IFR, Delta, country_dict, w_r=None, oth_r=None, sch_r=None, step=1):

    """
    This function integrates the system of differential equations
        :param e0 (float): initial fraction of E
        :param i0 (float): initial fraction of I
        :param r0 (float): initial fraction of R
        :param initial_date (datetime): initial date
        :param T (float): max time step
        :param R0 (float): basic reproductive number
        :param eps (float): incubation rate
        :param mu (float): recovery rate
        :param IFR (array): age-stratified infection fatality rate
        :param Delta (int): n. of days to observed death after the transition R -> D
        :param country_dict (dict): dictionary of country data
        :param w_r (array, default=None): work reduction parameters (Sept., Oct., Nov.) 
        :param oth_r (array, default=None): other_locations reduction parameters (Sept., Oct., Nov.)
        :param sch_r (array, default=None): school reduction parameters (Sept., Oct., Nov.)
        :param step (float, default=1): integration time step
        :return: returns the solution to the system dydt of differential equations

    Note:
        - age groups must be ordered sequentially (from youngest to oldest)
        - for each age group compartments must be ordered like this: S, E, I, R, D, DO
    """
    
 
    # get beta from R0 (R0 is always w.r.t to initial_date)
    beta = get_beta(R0, mu, update_contacts(country_dict, initial_date, w_r, oth_r, sch_r), country_dict["Nk"])
    
    # get number of age class and number of compartments
    nage  = len(country_dict["Nk"])

    # time
    t = 0

    # create solution array
    solution = np.zeros((nage, ncomp, T))
    
    # set initial conditions
    for a in range(nage):
        solution[a,0,0] = country_dict["Nk"][a] - int(country_dict["Nk"][a] * e0) - int(country_dict["Nk"][a] * i0) - int(country_dict["Nk"][a] * r0) # S
        solution[a,1,0] = int(country_dict["Nk"][a] * e0)         # E
        solution[a,2,0] = int(country_dict["Nk"][a] * i0)         # I
        solution[a,3,0] = int(country_dict["Nk"][a] * r0)         # R
        solution[a,4,0] = 0                                       # D
        solution[a,5,0] = 0                                       # DO
        
    # datetimes
    dates = [initial_date]

    # integrate
    for i in np.arange(1, T, 1):

        # advance timestamp
        dates.append(dates[-1] + timedelta(days=1))

        # update contacts
        C = update_contacts(country_dict, dates[-1], w_r, oth_r, sch_r)

        # integrate one step ahead
        sol = odeint(SEIRD_system, solution[:,:,i-1].ravel(), [t, t+step], args=(beta, eps, mu, Delta, C, country_dict["Nk"], IFR))
        
        # update the solution (note sol[0] is initial conditions)
        solution[:,:,i] = np.array(sol[1]).reshape(nage, ncomp)
        
        # advance time step
        t += step

    return solution, dates
