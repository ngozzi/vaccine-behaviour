# libraries
from scipy.integrate import odeint
import numpy as np
from datetime import datetime, timedelta
from numba import jit
from functions import get_beta

# n. of compartments, n. of age groups
ncomp = 20
nage  = 16


@jit
def BV_system(y, t, beta, eps, mu, omega, chi, f, r, vt, dt, alpha, gamma, VES, VEM, C, Nk, Delta, IFR, model):
    """
    This function defines the system of differential equations
        :param y (array): compartment values at time t
        :param t (float): current time step
        :param beta (float): infection rate
        :param eps (float): incubation rate
        :param mu (float): recovery rate
        :param omega (float): inverse of the prodromal phase length
        :param chi (float): relative infectivity of P, A infectious
        :param f (float): probability of being asymptomatic
        :param r (float): behavioural change parameter for non-compliant susceptibles
        :param vt (float): fraction of people already vaccinated
        :param dt (float): fraction of people died during last period
        :param alpha (float): behavioural change parameter for transition S -> S_NC (SV -> SV_NC)
        :param gamma (float): behavioural change parameter for transition S_NC -> S (SV_NC -> SV)
        :param VES (float): vaccine efficacy (on susceptibility)
        :param VEM (float): vaccine efficacy (on symptomaticity)
        :param C (matrix): contacts matrix
        :param Nk (array): number of individuals in different age groups
        :param Delta (int): n. of days to death after the transition I -> R
        :param IFR (array): age-stratified infection fatality rate
        :param model (string): vaccine_rate or constant_rate
        :return: returns the system dydt of differential equations

    Note:
        - age groups must be ordered sequentially (from youngest to oldest)
        - for each age group compartments must be ordered like this:
              S:0, SV:1, S_NC:2, SV_NC:3, E:4, P:5, A:6, I:7, RA:8, RI:9, D:10, DO: 11,
              E_V:12, P_V:13, A_V:14, I_V:15, RA_V:16, RI_V:17, D_V:18, DO_V: 19
    """

    # system
    dydt = []
    
    if model == "constant_rate":
        non_compl_rate = alpha
        compl_rate     = gamma
    elif model == "vaccine_rate":
        non_compl_rate = vt * alpha
        compl_rate     = dt * gamma  

    # iterate over age groups and define for each one the respective differential equations
    for age1 in range(nage):

        # first we compute the force of infection
        # beta * \sum_k' C_kk' * (I_k' + chi * (P_k' + A_k'))  / N_k'
        lambda_inf = 0
        for age2 in range(nage):
            # symptomatic contribution
            lambda_inf += beta * C[age1][age2] * (y[(ncomp * age2) + 7] + y[(ncomp * age2) + 15]) / Nk[age2]
            # pre-symptomatic and asymptomatic contribution
            lambda_inf += beta * chi * C[age1][age2] * (y[(ncomp * age2) + 5] + y[(ncomp * age2) + 6] + 
                                                        y[(ncomp * age2) + 13] + y[(ncomp * age2) + 14]) / Nk[age2]


        ### EQUATION FOR S ###
        dydt.append(-lambda_inf * y[(ncomp * age1) + 0]                        # S -> E (I, A, P)
                    -(1 - np.exp(-non_compl_rate)) * y[(ncomp * age1) + 0]     # S -> S_NC (-(vt * A))
                    +(1 - np.exp(-compl_rate)) * y[(ncomp * age1) + 2])        # S_NC -> S (+(dt * B))


        ### EQUATION FOR SV ###
        dydt.append(-(1 - VES) * lambda_inf * y[(ncomp * age1) + 1]            # SV -> E_V (I, A, P)
                    -(1 - np.exp(-non_compl_rate)) * y[(ncomp * age1) + 1]     # SV -> SV_NC (-(vt * A))
                    +(1 - np.exp(-compl_rate)) * y[(ncomp * age1) + 3])        # SV_NC -> SV (+(dt * B))


        ### EQUATION FOR S_NC ###
        dydt.append(-r * lambda_inf * y[(ncomp * age1) + 2]                    # S_NC -> E (I, A, P)
                    +(1 - np.exp(-non_compl_rate)) * y[(ncomp * age1) + 0]     # S -> S_NC (+(vt * A))
                    -(1 - np.exp(-compl_rate)) * y[(ncomp * age1) + 2])        # S_NC -> S (-(dt * B))


        ### EQUATION FOR SV_NC ###
        dydt.append(-r * (1 - VES) * lambda_inf * y[(ncomp * age1) + 3]        # SV_NC -> E_V (I)
                    +(1 - np.exp(-non_compl_rate)) * y[(ncomp * age1) + 1]     # SV -> SV_NC (+(vt * A))
                    -(1 - np.exp(-compl_rate)) * y[(ncomp * age1) + 3])        # SV_NC -> SV (-(dt * B))
        

        ### EQUATION FOR E ###
        dydt.append(+lambda_inf * y[(ncomp * age1) + 0]                        # S -> E (I, A, P)
                    +r * lambda_inf * y[(ncomp * age1) + 2]                    # S_NC -> E (I, A, P)
                    -eps * y[(ncomp * age1) + 4])                              # E -> P


        ### EQUATION FOR P ###
        dydt.append(+eps * y[(ncomp * age1) + 4]                    # E -> P
                    -omega * (1 - f) * y[(ncomp * age1) + 5]        # P -> I
                    -omega * f * y[(ncomp * age1) + 5])             # P -> A


        ### EQUATION FOR A ###
        dydt.append(+omega * f * y[(ncomp * age1) + 5]              # P -> A
                    -mu * y[(ncomp * age1) + 6])                    # A -> RA


        ### EQUATION FOR I ###
        dydt.append(+omega * (1 - f) * y[(ncomp * age1) + 5]        # P -> I
                    -mu * (1 - IFR[age1]) * y[(ncomp * age1) + 7]   # I -> RI
                    -mu * IFR[age1] * y[(ncomp * age1) + 7])        # I -> D
        
        
        ### EQUATION FOR RA ###
        dydt.append(+mu * y[(ncomp * age1) + 6])                    # A -> RA


        ### EQUATION FOR RI ###
        dydt.append(+mu * (1 - IFR[age1]) * y[(ncomp * age1) + 7])  # I -> RI


        ### EQUATION FOR D ###
        dydt.append(+mu * IFR[age1] * y[(ncomp * age1) + 7]         # I -> D
                    -(1 / Delta) * y[(ncomp * age1) + 10])          # D -> DO

        
        ### EQUATION FOR DO ###
        dydt.append(+(1 / Delta) * y[(ncomp * age1) + 10])          # D -> DO
        
        
        ### EQUATION FOR E_V ###
        dydt.append(+(1 - VES) * lambda_inf * y[(ncomp * age1) + 1]                # SV -> E_V (I, A, P)
                    +r * (1 - VES) * lambda_inf * y[(ncomp * age1) + 3]            # SV_NC -> E_V (I, A, P)
                    -eps * y[(ncomp * age1) + 12])                                 # E_V -> P_V
        
        
        ### EQUATION FOR P_V ###
        dydt.append(+eps * y[(ncomp * age1) + 12]                                  # E_V -> P_V
                    -omega * (1 - f) * (1 - VEM) * y[(ncomp * age1) + 13]          # P_V -> I_V
                    -omega * (1 - (1 - f) * (1 - VEM)) * y[(ncomp * age1) + 13])   # P_V -> A_V
        
        
        ### EQUATION FOR A_V ###
        dydt.append(+omega * (1 - (1 - f) * (1 - VEM)) * y[(ncomp * age1) + 13]    # P_V -> A_V
                    -mu * y[(ncomp * age1) + 14])                                  # A_V -> RA_V

        
        ### EQUATION FOR I_V ###
        dydt.append(+omega * (1 - f) * (1 - VEM) * y[(ncomp * age1) + 13]          # P_V -> I_V
                    -mu * (1 - IFR[age1]) * y[(ncomp * age1) + 15]                 # I_V -> RI_V
                    -mu * IFR[age1] * y[(ncomp * age1) + 15])                      # I_V -> D_V
        
        
        ### EQUATION FOR RA_V ###
        dydt.append(+mu * y[(ncomp * age1) + 14])                                  # A_V -> RA_V
        
        
        ### EQUATION FOR RI_V ###
        dydt.append(+mu * (1 - IFR[age1]) * y[(ncomp * age1) + 15])                # I_V -> RI_V
        
        
        ### EQUATION FOR D_V ###
        dydt.append(+mu * IFR[age1] * y[(ncomp * age1) + 15]                       # I_V -> D_V
                    -(1 / Delta) * y[(ncomp * age1) + 18])                         # D_V -> DO_V
        
        
        ### EQUATION FOR DO_V ###
        dydt.append(+(1 / Delta) * y[(ncomp * age1) + 18])                         # D_V -> DO_V
        

    return dydt


def integrate_BV(ics, T, R0, eps, mu, omega, chi, f, IFR, Delta, r, alpha, gamma, rV, VES, VEM, Cs, Nk, vaccination_strategy, model, dates, hesitancy, step=1):
    """
    This function integrates step by step the system defined previously.
        :param ics (matrix): initial conditions 
        :param T (float): max time step
        :param R0 (float): basic reproductive number
        :param eps (float): incubation rate
        :param mu (float): recovery rate
        :param omega (float): inverse of the prodromal phase length
        :param chi (float): relative infectivity of P, A infectious
        :param f (float): probability of being asymptomatic
        :param IFR (array): age-stratified infection fatality rate
        :param Delta (int): n. of days to death after the transition I -> R
        :param r (float): behavioural change parameter for non-compliant susceptibles
        :param alpha (float): behavioural change parameter for transition S -> S_NC
        :param gamma (float): behavioural change parameter for transition S_NC -> S
        :param rV (float): vaccination rate
        :param VES (float): vaccine efficacy (on susceptibility)
        :param VEM (float): vaccine efficacy (on symptomaticity)
        :param Cs (dict): dictionary of contacts matrices (keys are dates)
        :param Nk (array): number of individuals in different age groups
        :param vaccination_strategy (string): old_first, 20_49_first, homogeneous
        :param model (string): vaccine_rate or constant_rate
        :param dates (array): list of dates
        :param hesitancy (float): fraction of people refusing vaccine
        :param step (float, default=1): time step
        :return: returns the solution to the system dydt of differential equations

    Note:
        - age groups must be ordered sequentially (from youngest to oldest)
        - for each age group compartments must be ordered like this: 
                      S:0, SV:1, S_NC:2, SV_NC:3, E:4, P:5, A:6, I:7, RA:8, RI:9, D:10, DO: 11,
                      E_V:12, P_V:13, A_V:14, I_V:15, RA_V:16, RI_V:17, D_V:18, DO_V: 19
        - the timestep by timestep trick is needed to properly update vt and dt
    """

    # create solution array
    solution = np.zeros((nage, ncomp, T))

    # get beta
    beta = get_beta(R0, mu, chi, omega, f, Cs[dates[0]], Nk)

    # total population
    Ntot = np.sum(Nk)

    # set initial conditions
    solution[:,:,0] = ics

    V = 0     # tot n. of vaccinated so far
    t = 0     #  current time
    Vt = []   # time series of new vaccinated
    vaccines_wasted = 0  # vaccines wasted on not susceptible

    # integrate
    for i in np.arange(1, T, 1):
        
        # set non-negative constraints
        for age in range(nage):
            for comp in range(ncomp):
                if solution[age, comp, i - 1] < 0:
                    solution[age, comp, i - 1] = 0

        # update fraction of vaccinated and dead
        vt = V / Ntot
        if i >= 2:
            dt = 100000 * ((solution.sum(axis=0)[11][i - 1] -  solution.sum(axis=0)[11][i - 2]) + 
                           (solution.sum(axis=0)[19][i - 1] -  solution.sum(axis=0)[19][i - 2])) / np.sum(Nk)
        else: 
            dt = 0.0

        # n. of people to vaccinate in this time step, update
        V_S, V_S_NC, v_wasted = get_vaccinated(rV, Nk, vaccination_strategy, solution[:, :, i - 1].ravel(), vaccines_wasted, hesitancy)
        
        # update wasted vaccines
        vaccines_wasted += v_wasted
        
        for age in range(nage):
            solution[age, 1, i - 1] += V_S[age]     # SV
            solution[age, 0, i - 1] -= V_S[age]     # S
            solution[age, 3, i - 1] += V_S_NC[age]  # SV_NC
            solution[age, 2, i - 1] -= V_S_NC[age]  # S_NC
            
        # update contacts
        C = Cs[dates[i]]

        # integrate one step ahead
        sol = odeint(BV_system, solution[:, :, i - 1].ravel(), [t, t + step], args=(beta, eps, mu, omega, chi, f, r, vt, dt, alpha, 
                                                                                    gamma, VES, VEM, C, Nk, Delta, IFR, model))
        
        # update the solution (note sol[0] is the system at time t, we want t+step)
        solution[:, :, i] = np.array(sol[1]).reshape(nage, ncomp)
        
        # update number of vaccinated
        Vt.append(np.sum(V_S) + np.sum(V_S_NC))
        V += Vt[-1]

        # advance time step
        t += step

    return solution, Vt, vaccines_wasted

@jit
def get_vaccinated(rV, Nk, vaccination_strategy, y, vaccines_wasted, hesitancy):
    """
    This functions compute the n. of S individuals that will receive a vaccine in the next step
        :param rV (float): vaccination rate
        :param Nk (array): number of individuals in different age groups
        :param vaccination_strategy (string): old_first, 20_49_first, homogeneous
        :param y (array): compartment values at time t
        :param vaccines_wasted (float): number of vaccines already wasted
        :param hesitancy (float): fraction of people refusing vaccine
        :return: returns the two arrays of n. of vaccinated in different age groups for S and S_NC in the next step
    """

    # list of n. of vaccinated in each age group for S and S_NC in this step
    V_S, V_S_NC = np.zeros(nage), np.zeros(nage)
    
    # vaccines wasted in this step 
    vaccines_wasted_step = 0

    # tot n. of vaccine available this step
    totV = rV * np.sum(Nk)
    
    # rescale doses available
    S, nonS = 0, 0
    for age in range(nage):
        nonS += (y[(ncomp * age) + 4] + y[(ncomp * age) + 5] + y[(ncomp * age) + 6] + y[(ncomp * age) + 8])
        S += (y[(ncomp * age) + 0] + y[(ncomp * age) + 2])
        
    totV_eff = totV * (S / (S + nonS - vaccines_wasted))
    #totV_eff = totV
    vaccines_wasted_step = totV - totV_eff
        
    # homogeneous vaccination
    if vaccination_strategy == "homogeneous":

        # tot people that can receive the vaccines
        den = 0
        for age in range(nage):
            den += (y[(ncomp * age) + 0] + y[(ncomp * age) + 2])

        # all vaccinated
        if den <= 1:
            return np.zeros(nage), np.zeros(nage), 0

        # distribute vaccine homogeneously
        for age in range(nage):
            
            # check hesitancy
            #if y[(ncomp * age) + 0] > Nk[age] * hesitancy * (y[(ncomp * age) + 0] / (y[(ncomp * age) + 0] + y[(ncomp * age) + 2])):
            if y[(ncomp * age) + 0] + y[(ncomp * age) + 2] > Nk[age] * hesitancy:
                
                V_S[age] = totV_eff * y[(ncomp * age) + 0] / den
                V_S_NC[age] = totV_eff * y[(ncomp * age) + 2] / den
                
                # check we are not exceeding the tot n. of susceptibles left
                if V_S[age] > y[(ncomp * age) + 0]:
                    V_S[age] = y[(ncomp * age) + 0]
                    
                if V_S_NC[age] > y[(ncomp * age) + 2]:
                    V_S_NC[age] = y[(ncomp * age) + 2]    
                         
    

    # prioritize elderly in vaccination
    elif vaccination_strategy == "old_first":
        
        # from older to younger
        for age in np.arange(15, -1, -1):
            
            # left vaccine for this step
            left_V = totV_eff - np.sum(V_S) - np.sum(V_S_NC) 
            
            #  check there are still vaccines this step
            if left_V < 1:
                return V_S, V_S_NC, vaccines_wasted_step
            
            else:
                # assign to compliant / non-compliant homogeneously
                den = (y[(ncomp * age) + 0] + y[(ncomp * age) + 2])
                
                if den > Nk[age] * hesitancy:
                    V_S[age] = left_V * y[(ncomp * age) + 0] / den
                    V_S_NC[age] = left_V * y[(ncomp * age) + 2] / den
                    
                    # check we are not exceeding the tot n. of susceptibles left
                    if V_S[age] > y[(ncomp * age) + 0]:
                        V_S[age] = y[(ncomp * age) + 0]
                    if V_S_NC[age] > y[(ncomp * age) + 2]:
                        V_S_NC[age] = y[(ncomp * age) + 2]
                        

    elif vaccination_strategy == "20-49_first":
        # first vaccinate 20-49 homogeneously
        den = 0
        for age in [4, 5, 6, 7, 8, 9]:
            den += (y[(ncomp * age) + 0] + y[(ncomp * age) + 2])

        # not all vaccinated
        if den > 1:

            # distribute vaccine homogeneously among 20-49
            for age in [4, 5, 6, 7, 8, 9]:
                V_S[age] = totV_eff * y[(ncomp * age) + 0] / den
                V_S_NC[age] = totV_eff * y[(ncomp * age) + 2] / den

                # check we are not exceeding the tot n. of susceptibles left
                if V_S[age] > y[(ncomp * age) + 0]:
                    V_S[age] = y[(ncomp * age) + 0]
                if V_S_NC[age] > y[(ncomp * age) + 2]:
                    V_S_NC[age] = y[(ncomp * age) + 2]
                    
        # left vaccine for this step
        left_V = totV_eff - np.sum(V_S) - np.sum(V_S_NC)
        #  check there are still vaccines this step
        if left_V < 1:
            return V_S, V_S_NC, vaccines_wasted_step
        else:
            den = 0
            # vaccinate all other 
            for age in [0, 1, 2, 3, 10, 11, 12, 13, 14, 15]:
                den += (y[(ncomp * age) + 0] + y[(ncomp * age) + 2])

            # not all vaccinated
            if den > 1:

                # distribute vaccine homogeneously among 20-49
                for age in [0, 1, 2, 3, 10, 11, 12, 13, 14, 15]:
                    V_S[age] = left_V * y[(ncomp * age) + 0] / den
                    V_S_NC[age] = left_V * y[(ncomp * age) + 2] / den

                    # check we are not exceeding the tot n. of susceptibles left
                    if V_S[age] > y[(ncomp * age) + 0]:
                        V_S[age] = y[(ncomp * age) + 0]
                    if V_S_NC[age] > y[(ncomp * age) + 2]:
                        V_S_NC[age] = y[(ncomp * age) + 2]

    return V_S, V_S_NC, vaccines_wasted_step

