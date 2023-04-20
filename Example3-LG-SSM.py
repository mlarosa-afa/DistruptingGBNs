import numpy as np
from scipy.stats import invgamma, norm, invwishart, multivariate_normal
import functions
from docplex.mp.model import Model
from gb_SAA import gb_SAA
from wb_attack import whitebox_attack

np.random.seed(100)
def LGSSM_Generate(T, epsilon_var=None, delta_var=None, mu_state_init=None, var_state_init=None, seed=23):
    np.random.seed(seed)
    if epsilon_var is None:
        epsilon_var = invgamma.rvs(1, loc = 0, scale = 1, size = 4) #[1, 1,0.5,0.5] #Error st dev for transtition model - SAMPLE FOR PROBLEM GB
    if delta_var is None:
        delta_var = invgamma.rvs(1, loc = 0, scale = 1, size = 2)#[0.2, 0.2]   #Error st dev for observation model - SAMPLE FOR PROBLEM GB
    if mu_state_init is None:
        mu_state_init = norm.rvs(loc = 1, scale = 2, size=4)#[0,0,1,1] # SAMPLE FOR PROBLEM GB
    if var_state_init is None:
        var_state_init = invgamma.rvs(1, loc = 0, scale = 1, size = 4)#[0.5,0.5,0.5,0.5] # SAMPLE FOR PROBLEM GB
    Delta_t = 1

    #Build matrix of beta coefficients
    beta = np.zeros((6 * (T + 1), 6 * (T + 1))) #Initialize to zero matrix
    beta[4][0] = 1
    beta[5][1] = 1
    for t in range(1, T+1):
        #Coef for state variables
        beta[6*t][(6*(t-1)):(6*(t-1)+6)] = 1,0,Delta_t,0,0,0
        beta[6*t+1][(6*(t-1)):(6*(t-1)+6)] = 0,1,0,Delta_t,0,0
        beta[6*t+2][(6*(t-1)):(6*(t-1)+6)] = 0,0,1,0,0,0
        beta[6*t+3][(6*(t-1)):(6*(t-1)+6)] = 0,0,0,1,0,0
        #Coef for emissions(obs)
        beta[6*t+4, 6*t] = 1
        beta[6*t+5, 6*t+1] = 1


    mu = np.zeros((6*(T+1), 1)) #Initialize

    #Find mu of joint PDF -- loop thru Koller's recursion
    for t in range(T+1):
      if t == 0:
        for i in range(5): #Update all state/obs means
            if i < 4:
              mu[i-1] = mu_state_init[i-1] #Position and velocities
            else:
              mu[i-1] = mu[i-4-1] #Sensor readings

      else: # when t>0
        for i in range(6): #Update all state/obs means
         if i <= 1:
            mu[6*t+i] = mu[6*(t-1)+i] + Delta_t*mu[6*(t-1)+(i+2)] #Position
         elif i == 2 or i == 3:
            mu[6*t+i] = mu[6*(t-1)+i] #Velocities
         else:
            mu[6*t+i] = mu[6*t+(i-4)] #Sensor readings

    Sigma = np.zeros((6 * (T + 1), 6 * (T + 1))) #Initialize to zero matrix

    for i in range((6*(T+1))):
      for j in range(i+1):
        #Check if i=j and t=0
        if i == j and  i <= 5:
          if i<= 3:
            Sigma[i,j] = var_state_init[i]
          else:
            Sigma[i,j] = delta_var[i-4] + np.transpose(beta[i,:(j-1)]) @ Sigma[:(j-1), :(j-1)] @ beta[i,:(j-1)]

        #Check if i=j and t>0
        if i == j and i>=6:
          if i%6<=3 and i%6>=0:
            Sigma[i,j] = epsilon_var[i%6] + np.transpose(beta[i,:(j-1)]) @ Sigma[:(j-1), :(j-1)] @ beta[i,:(j-1)]
          else:
            if i%6 == 4:
                Sigma[i,j] = delta_var[1] + np.transpose(beta[i,:(j-1)]) @ Sigma[:(j-1), :(j-1)] @ beta[i,:(j-1)]
            else:
                Sigma[i,j] = delta_var[0] + np.transpose(beta[i,:(j-1)]) @ Sigma[:(j-1), :(j-1)] @ beta[i,:(j-1)]

        #Check if i!=j and i>5 (i.e., we are at the first y10 value - at least)
        if i != j and i>=3:
            Sigma[i,j] = np.transpose(Sigma[j, :(i-1)]) @ beta[i, :(i-1)]
            Sigma[j,i] = Sigma[i,j]

    return Sigma, mu

mode = 2# int(input("1) Whitebox Attack\n2) Graybox - Sample Average Approximation\nSelected Mode: "))
if mode == 1:
    U_1 = float(input("Enter U_1: "))
    U_2 = 1 - U_1
    print("Caluclated weight 2 as ", U_2)
    evidence = [0.07, 0, 1.16, .68, 2.25, 1, 3.11, 1.32, 4.05, 2.11, 5.18, 2.36, 6.24, 3.2, 6.81, 3.56, 7.77, 3.82,
                9.15, 4.46]  # multiples of 6 starting with 4 and 5 (zero indexed)
    #build LG-SSM based off of evidence
    T = int(len(evidence) / 2)
    MVG_Sigma, MVG_mu = LGSSM_Generate(T)

    ev_vars = []  # multiples of 6 starting with 4 and 5 (zero indexed)
    i = 4
    while i < T * 6:
        ev_vars.append(i)
        ev_vars.append(i+1)
        i = i + 6

    b_concave, b_convex, solutionset, qm = whitebox_attack(MVG_Sigma, MVG_mu, ev_vars, evidence, U_1, U_2, risk_tolerance=.5)
    print("The interesting range of weight 1 ranges from ", b_concave, " to ", b_convex)
    print("Solutions that do not appear are zero.")

if mode == 2:

    U_1 = float(input("Enter U 1: "))
    testVal = [0.1, 0.3, 0.5, 0.7, 0.9]
    for k in testVal:
        U_1 = k
        U_2 = 1 - U_1
        print("u_1 = ", k)
        numSamples = 100 #int(input("Enter number of Samples: "))
        evidence = [0.07, 0, 1.16, .68, 2.25, 1, 3.11, 1.32, 4.05, 2.11, 5.18, 2.36, 6.24, 3.2, 6.81, 3.56, 7.77, 3.82,
                    9.15, 4.46]  # multiples of 6 starting with 4 and 5 (zero indexed)
        # build LG-SSM based off of evidence
        T = int(len(evidence) / 2)
        MVG_Sigma, MVG_mu = LGSSM_Generate(T)

        ev_vars = []  # multiples of 6 starting with 4 and 5 (zero indexed)
        i = 4
        while i < T * 6:
            ev_vars.append(i)
            ev_vars.append(i + 1)
            i = i + 6

        cov_samples = []
        mu_samples = []
        for i in range(numSamples):
            cov_sample, mu_sample = LGSSM_Generate(T)
            cov_samples.append(cov_sample)
            mu_samples.append(mu_sample)

        solution = gb_SAA(cov_samples, mu_samples, ev_vars, evidence, U_1, U_2, risk_tolerance=.75)
        print("Solutions that do not appear are zero.")
        print(solution)
