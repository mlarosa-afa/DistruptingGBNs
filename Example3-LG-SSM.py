import numpy as np
from scipy.stats import invgamma, norm, invwishart, multivariate_normal
import functions
from docplex.mp.model import Model

#FIX check_bounds (multimodes to handle certain amount off position) - Based on true value
#add SAA
#remove random evidence
#Add WB
#add SDG on Zillow (only)

np.random.seed(100)
def LGSSM_Generate(epsilon_var=None, delta_var=None, mu_state_init=None, var_state_init=None):
    T = 1
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


W_1 = float(input("Enter U 1: "))
W_2 = 1 - W_1

evidence_vars = [4,5,10,11]
unobserved_vars = [0,1,2,3,6,7,8,9]
observed_vals = [0,0,1,1]

print("Please select a method:\n\t1.AdaGrad\n\t2.RMSProp\n\t3.Adam")
method = int(input("Method:"))
LEARN_RATE = float(input("Learning Rate: "))

print("Z_1_t, Z_2_t, Z_1_t+1, Z_2_t+1")
#input vector of sensor obs
solution = np.array([0, 0, 0, 0])
prev_solution = np.array([10000, 10000, 10000, 10000])
v = 0
t = 1
m = 0

#number of iters
mode = int(input("mode: "))
if mode == 1:

    vals = functions.params_from_sample(sample_cov, sample_Mu, evidence_vars, unobserved_vars, observed_vals)
    Dmat = ((W_1 * vals.Q) - (W_2 * vals.K_prime))
    Dvec = W_1 * np.transpose(vals.vT) + 2 * W_2 * np.matmul(vals.K_prime, vals.u_prime)

    qm = Model('DistruptionGBN')
    z_DV = qm.continuous_var_matrix(1, 4, name="Z_DV", lb=-3, ub=3)  # DV for decision variable

    # Solve normalized problem
    W_1 = U_1 / Phi_opt1
    W_2 = U_2 / Phi_opt2

    Dmat = (W_1 * v.Q) - (W_2 * v.K_prime)
    Dvec = W_1 * np.transpose(v.vT) + 2 * W_2 * np.matmul(v.K_prime, v.u_prime)
    obj_fn = (list(z_DV.values()) @ Dmat @ list(z_DV.values())) + (Dvec @ list(z_DV.values()))
    qm.set_objective("max", obj_fn)

    qm.parameters.optimalitytarget.set(3)
    # qm.parameters.optimalitytarget.set(2)

    solutionset = qm.solve()
    qm.print_solution()

if mode == 2:
    while abs(np.linalg.norm(solution-prev_solution)) > .001:
        sample_cov, sample_Mu = LGSSM_Generate()

        vals = functions.params_from_sample(sample_cov, sample_Mu, evidence_vars, unobserved_vars, observed_vals)
        Dmat = ((W_1 * vals.Q) - (W_2 * vals.K_prime))
        Dvec = W_1 * np.transpose(vals.vT) + 2 * W_2 * np.matmul(vals.K_prime, vals.u_prime)
        prev_solution = solution

        if method == 1:
            solution, v = functions.adaGrad(lambda z: (Dmat + Dmat.transpose()) @ z + (Dvec), lambda z: z <= 3, solution, LEARN_RATE, v = v)
        elif method == 2:
            solution, v = functions.RMSProp(lambda z: (Dmat + Dmat.transpose()) @ z + (Dvec), lambda z: z <= 3, solution, LEARN_RATE, v=v)
        elif method == 3:
            solution, v, m = functions.adam(lambda z: (Dmat + Dmat.transpose()) @ z + (Dvec), lambda z: z <= 3, solution, learn_rate=LEARN_RATE, t = t, v = v, m = m)
            t = t + 1
        print(solution)

