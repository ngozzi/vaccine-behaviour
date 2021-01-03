# libraries
from scipy.integrate import odeint
import numpy as np
from datetime import datetime, timedelta
from functions import update_contacts, get_beta

# n. of compartments
ncomp = 9
 

def BV_system(y, t, beta, eps, mu, Delta, r, vt, dt, A, B, rV, VE, tV0, maxV, C, Nk, V_S, V_S_NC, IFR):
    
    """
    This function defines the system of differential equations
        :param y (array): compartment values at time t
        :param t (float): current time step
        :param beta (float): infection rate
        :param eps (float): incubation rate
        :param mu (float): recovery rate
        :param Delta (float): inverse of the rate of D -> DO
        :param r (float): behavioural change parameter for non-compliant susceptibles
        :param vt (float): fraction of people already vaccinated
        :param dt (float): fraction of people died during last period
        :param A (float): behavioural change parameter for transition S -> S_NC (SV -> SV_NC)
        :param B (float): behavioural change parameter for transition S_NC -> S (SV_NC -> SV)
        :param rV (float): vaccination rate
        :param VE (float): vaccine efficacy (on susceptibility)
        :param tV0 (float): start of vaccination
        :param maxV (float): max fraction of people that can be vaccinated
        :param C (matrix): contacts matrix
        :param Nk (array): number of individuals in different age groups
        :param V_S (array): number of S individuals to be vaccinated in this step 
        :param V_S_NC (array): number of S_NC individuals to be vaccinated in this step
        :param IFR (array): age-stratified infection fatality rate
        :return: returns the system dydt of differential equations

    Note:
        - age groups must be ordered sequentially (from youngest to oldest)
        - for each age group compartments must be ordered like this: S, SV, S_NC, SV_NC, E, I, R, D, DO
    """
    
    # number of age groups
    nage = len(C)

    # system
    dydt = []

    # iterate over age groups and define for each one the respective differential equations
    for age1 in range(nage):

        # first we compute all interaction terms with other age groups
        S_to_E_I = 0
        for age2 in range(nage):
            S_to_E_I += C[age1][age2] * y[(ncomp * age2) + 5] / Nk[age2]   # \sum_k' C_kk' I_k'  / N_k'
        
        
        ### EQUATION FOR S ###            
        dydt.append(-beta * S_to_E_I * y[(ncomp * age1) + 0]                    # S -> E (I)
                    -(vt * A) * y[(ncomp * age1) + 0]                           # S -> S_NC
                    +(dt * B) * y[(ncomp * age1) + 2]                           # S_NC -> S
                    -V_S[age1])                                                 # S -> SV

        ### EQUATION FOR SV ###
        dydt.append(-(1 - VE) * beta * S_to_E_I * y[(ncomp * age1) + 1]         # SV -> E (I)
                    -(vt * A) * y[(ncomp * age1) + 1]                           # SV -> SV_NC
                    +(dt * B) * y[(ncomp * age1) + 3]                           # SV_NC -> SV
                    +V_S[age1])                                                 # S  -> SV
        
        ### EQUATION FOR S_NC ###
        dydt.append(-r * beta * S_to_E_I * y[(ncomp * age1) + 2]                # S_NC -> E (I)
                    +(vt * A) * y[(ncomp * age1) + 0]                           # S -> S_NC
                    -(dt * B) * y[(ncomp * age1) + 2]                           # S_NC -> S
                    -V_S_NC[age1])                                              # S_NC -> SV_NC
        
        ### EQUATION FOR SV_NC ###
        dydt.append(-r * (1 - VE) * beta * S_to_E_I * y[(ncomp * age1) + 3]     # SV_NC -> E (I)
                    +(vt * A) * y[(ncomp * age1) + 1]                           # SV -> SV_NC
                    -(dt * B) * y[(ncomp * age1) + 3]                           # SV_NC -> SV
                    +V_S_NC[age1])                                              # S_NC  -> SV_NC

        ### EQUATION FOR E ###
        dydt.append(+beta * S_to_E_I * y[(ncomp * age1) + 0]                    # S -> E (I)
                    +(1 - VE) * beta * S_to_E_I * y[(ncomp * age1) + 1]         # SV -> E (I)
                    +r * beta * S_to_E_I * y[(ncomp * age1) + 2]                # S_NC -> E (I)
                    +r * (1 - VE) * beta * S_to_E_I * y[(ncomp * age1) + 3]     # SV_NC -> E (I)
                    -eps * y[(ncomp * age1) + 4])                               # E -> I

        ### EQUATION FOR I ###
        dydt.append(+eps * y[(ncomp * age1) + 4]                                # E -> I
                    -mu * (1 - IFR[age1]) * y[(ncomp * age1) + 5]               # I -> R
                    -mu * IFR[age1] * y[(ncomp * age1) + 5])                    # I -> D
        
        ### EQUATION FOR R ###
        dydt.append(+mu * (1 - IFR[age1]) * y[(ncomp * age1) + 5])              # I -> R
        
        ### EQUATION FOR D ### 
        dydt.append(+mu * IFR[age1] * y[(ncomp * age1) + 5]                     # I -> D
                    -(1 / Delta) * y[(ncomp * age1) + 7])                       # D -> DO
        
        ### EQUATION FOR DO ###
        dydt.append(+(1 / Delta) * y[(ncomp * age1) + 7])                       # D -> DO
        
    return dydt



def get_vaccinated_next_step(nage, t, tV0, vt, maxV, rV, Nk, vaccination_by_age, y):
    
    """
    This functions compute the n. of S individuals that will receive a vaccine in the next step 
        :param nage (int): number of age groups
        :param ncomp (int): number of compartments
        :param t (float): current time step
        :param tV0 (float): start of vaccination
        :param vt (float): fraction of people already vaccinated
        :param maxV (float): max fraction of people that can be vaccinated
        :param rV (float): vaccination rate
        :param Nk (array): number of individuals in different age groups    
        :param vaccination_by_age (bool): if True 60+ are vaccinated first, otherwise vacc. is homogeneous
        :param y (array): compartment values at time t
        :return: returns the two arrays of n. of vaccinated in different age groups for S and S_NC in the next step 
    """
    
    # list of n. of vaccinated in each age group for S and S_NC in this step 
    V_S, V_S_NC = np.zeros(nage), np.zeros(nage)  
    
    # if vaccination started and not max % has been reached already
    if t >= tV0 and vt < maxV: 
        
        # tot n. of vaccine available this step 
        totV = rV * np.sum(Nk)  
        
        # check we are not exceeding max 
        if (totV / np.sum(Nk) + vt) > maxV: 
            totV = (maxV - vt) * np.sum(Nk)
        
        # homogeneous vaccination
        if vaccination_by_age == False:
            
            den = 0
            for age in range(nage):
                den += (y[(ncomp * age) + 0] + y[(ncomp * age) + 2])
                
            # all vaccinated
            if den <= 1: 
                return np.zeros(nage), np.zeros(nage)
                
            # distribute vaccine homogeneously
            for age in range(nage):
                V_S[age] = totV * y[(ncomp * age) + 0] / den
                V_S_NC[age] = totV * y[(ncomp * age) + 2] / den
                
                # check we are not exceeding the tot n. of susceptibles left
                if V_S[age] > y[(ncomp * age) + 0]:
                    V_S[age] = y[(ncomp * age) + 0]
                if V_S_NC[age] > y[(ncomp * age) + 2]:
                    V_S_NC[age] = y[(ncomp * age) + 2]
                 
        # prioritize elderly in vaccination
        else:
            # from older to younger
            for age in np.arange(15, -1, -1):
                # left vaccine for this step
                left_V = totV - np.sum(V_S) - np.sum(V_S_NC)
                # check there are still vaccines this step
                if left_V < 1:
                    return V_S, V_S_NC
                else:
                    # assign to compliant / non-compliant homogeneously
                    den = (y[(ncomp * age) + 0] + y[(ncomp * age) + 2])
                    if den > 1:
                        #if y[(ncomp * age) + 0] > 0:
                        V_S[age] = left_V * y[(ncomp * age) + 0] / den
                        V_S_NC[age] = left_V * y[(ncomp * age) + 2] / den
                        # check we are not exceeding the tot n. of susceptibles left
                        if V_S[age] > y[(ncomp * age) + 0]:
                            V_S[age] = y[(ncomp * age) + 0]
                        if V_S_NC[age] > y[(ncomp * age) + 2]:
                            V_S_NC[age] = y[(ncomp * age) + 2]

        
        #else:
            # first vaccinate 60+
            #den = 0
            #for age in np.arange(12, 16, 1):
            #    den += (y[(ncomp * age) + 0] + y[(ncomp * age) + 2])
            #    
            #for age in np.arange(12, 16, 1):
            #    V_S[age] = totV * y[(ncomp * age) + 0] / den
            #    V_S_NC[age] = totV * y[(ncomp * age) + 2] / den
            #    
            #    # check we are not exceeding the tot n. of susceptibles left
            #    if V_S[age] > y[(ncomp * age) + 0]:
            #        V_S[age] = y[(ncomp * age) + 0]
            #    if V_S_NC[age] > y[(ncomp * age) + 2]:
            #        V_S_NC[age] = y[(ncomp * age) + 2]

            ## we give the vaccines left to the other age groups (in decreasing order)
            #left_V = totV - np.sum(V_S) - np.sum(V_S_NC)
            #den = 0
            #for age in np.arange(0, 12, 1):
            #    den += (y[(ncomp * age) + 0] + y[(ncomp * age) + 2])
            #    
            #for age in np.arange(0, 12, 1):
            #    V_S[age] = left_V * y[(ncomp * age) + 0] / den
            #    V_S_NC[age] = left_V * y[(ncomp * age) + 2] / den
            #    
                # check we are not exceeding the tot n. of susceptibles left
            #    if V_S[age] > y[(ncomp * age) + 0]:
            #        V_S[age] = y[(ncomp * age) + 0]
            #    if V_S_NC[age] > y[(ncomp * age) + 2]:
            #        V_S_NC[age] = y[(ncomp * age) + 2]
                    
    return V_S, V_S_NC



def integrate_BV(y0, initial_date, T, beta, eps, mu, IFR, Delta, r, A, B, rV, VE, tV0, maxV, country_dict, scenario, vaccination_by_age, step=1):
    
    """
    This function integrates step by step the system defined previously. 
        :param y0 (array): initial conditions
        :param initial_date (datetime): initial date
        :param T (float): max time step
        :param R0 (float): basic reproductive number
        :param eps (float): incubation rate
        :param mu (float): recovery rate
        :param IFR (array): age-stratified infection fatality rate
        :param Delta (int): n. of days to death after the transition I -> R
        :param r (float): behavioural change parameter for non-compliant susceptibles
        :param A (float): behavioural change parameter for transition S -> S_NC
        :param B (float): behavioural change parameter for transition S_NC -> S
        :param rV (float): vaccination rate
        :param VE (float): vaccine efficacy (on susceptibility)
        :param tV0 (float): start of vaccination
        :param maxV (float): max fraction of people that can be vaccinated
        :param country_dict (dict): dictionary of country data
        :param scenario (int): number of reopening scenario
        :param vaccination_by_age (bool): if True 60+ are vaccinated first, otherwise vacc. is homogeneous
        :param step (float, default=1): time step
        :return: returns the solution to the system dydt of differential equations
        
    Note:
        - age groups must be ordered sequentially (from youngest to oldest)
        - for each age group compartments must be ordered like this: S, SV, S_NC, SV_NC, E, I, R, D, DO
        - the timestep by timestep trick is needed to properly update vt and dt
    """
    
    # get beta from R0 (R0 is always w.r.t to initial_date)
    #beta = get_beta(R0, mu, update_contacts(country_dict, datetime(2020,9,1), scenario=scenario), country_dict["Nk"])
    
    # get number of age class and number of compartments
    nage  = len(country_dict["Nk"])
    
    # create solution array (we add two compartments, for deaths of vaccinated and not)
    solution  = np.zeros((nage, ncomp, T))
    
    # set initial conditions
    solution[:,:,0] = y0 #np.array(y0).reshape(nage, ncomp)
    
    V  = 0   # tot n. of vaccinated so far
    t  = 0   # current time
    Vt = []  # time series of new vaccinated
    dates = [initial_date] # datetimes
    
    # integrate
    for i in np.arange(1, T, 1):
        
        # advance timestamp
        dates.append(dates[-1] + timedelta(days=1))
        
        # update contacts
        C = update_contacts(country_dict, dates[-1], scenario=scenario)
        
        # update fraction of vaccinated and dead
        vt = V / np.sum(country_dict["Nk"])
        if i >= 2:
            dt = np.sum(solution.sum(axis=0)[8][i-1] - solution.sum(axis=0)[8][i-2])
        else:
            dt = 0.0
        
        # n. of people to vaccinate in the next time step
        V_S, V_S_NC = get_vaccinated_next_step(nage, t, tV0, vt, maxV, rV, country_dict["Nk"], vaccination_by_age, solution[:,:,i-1].ravel())
        
        # integrate one step ahead
        sol = odeint(BV_system, solution[:,:,i-1].ravel(), [t, t+step], args=(beta, eps, mu, Delta, r, vt, dt, A, B, rV, VE, tV0, maxV, C, country_dict["Nk"], V_S, V_S_NC, IFR))

        # update the solution (note sol[0] is the system at time t, we want t+step)
        solution[:,:,i] = np.array(sol[1]).reshape(nage, ncomp)
        
        for a in range(nage):
            for c in range(ncomp):
                if solution[a,c,i] < 1:
                    solution[a,c,i] = 0.0

        # update number of vaccinated
        Vt.append(np.sum(V_S) + np.sum(V_S_NC))
        V += Vt[-1]

        # advance time step
        t += step
        
    return solution, Vt, dates