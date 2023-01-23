import numpy as np
import random
from scipy.stats import invwishart
import math

def vals_from_priors(MVG_Sigma, MVG_Mu, NUM_EVIDENCE_VARS = 1):
    Lambda_dot = np.linalg.inv(MVG_Sigma)  #Precision matrix of joint evidence distribution
    eta_dot = np.matmul(Lambda_dot, MVG_Mu)
    Xi_dot = -0.5 * np.matmul(np.matmul(np.transpose(MVG_Mu), Lambda_dot), MVG_Mu) - math.log((2*math.pi)**(MVG_Mu.size/2) * (np.linalg.det(MVG_Sigma) ** 0.5))

    random.seed(2023)
    #Generate Random evidence
    evidence_vars = random.sample(range(MVG_Mu.shape[0]), NUM_EVIDENCE_VARS)
    evidence_vars.sort()

    observed_vals = list()
    for j in evidence_vars:
        observed_vals.append(np.random.normal(MVG_Mu[j], MVG_Sigma[j, j], 1))

    unobserved_vars = list(set(list(range(MVG_Mu.size))) - set(evidence_vars))
    # Lambda Subsets  - Form Block Matrices from Joint Covariance Matrix
    Lambda_yy = np.delete(Lambda_dot, evidence_vars, 1)
    Lambda_yy = np.delete(Lambda_yy, evidence_vars, 0)
    #Lambda_yz = Lambda_dot[!(vars % in% evidence_vars), (vars % in% evidence_vars)]
    Lambda_yz = np.delete(Lambda_dot, unobserved_vars, 1)
    Lambda_yz = np.delete(Lambda_yz, evidence_vars, 0)
    Lambda_zy = np.transpose(Lambda_yz)
    Lambda_zz = np.delete(Lambda_dot, unobserved_vars, 1)
    Lambda_zz = np.delete(Lambda_zz, unobserved_vars, 0)

    #eta Subsets
    eta_y = np.delete(eta_dot, evidence_vars)
    eta_z = np.delete(eta_dot, unobserved_vars)

    # Conditional Distribution
    K = Lambda_yy
    h = eta_y - np.matmul(Lambda_yz, observed_vals)
    g = Xi_dot + np.matmul(np.transpose(eta_z), observed_vals) - (0.5 * np.matmul(np.matmul(np.transpose(observed_vals), Lambda_zz), observed_vals))

    u_dot = np.matmul(np.linalg.inv(K), h)

    #KL Divergence
    K_prime = Lambda_zz - np.matmul(np.matmul(Lambda_zy,np.linalg.inv(Lambda_yy)),Lambda_yz)
    h_prime = eta_z - np.matmul(Lambda_zy, np.matmul(np.linalg.inv(Lambda_yy), eta_y))

    S_prime = np.linalg.inv(K_prime)
    u_prime = np.matmul(S_prime, h_prime)

    Q = np.matmul(np.matmul(np.transpose(Lambda_yz), np.linalg.inv(K)), Lambda_yz)
    vT = 2 * (np.matmul(np.transpose(u_dot), Lambda_yz) - np.matmul(np.matmul(np.transpose(eta_y), np.linalg.inv(K)), Lambda_yz))
    c = np.matmul(np.matmul(np.transpose(u_dot), K), u_dot) - 2 * np.matmul(np.transpose(eta_y), u_dot) + np.matmul(np.transpose(eta_y), np.matmul(np.linalg.inv(K), eta_y))
    return Q, Lambda_zz

#https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
#Assuming Symetric (Is that okay?)
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

if __name__ == '__main__':
    maxdim = int(input("Max Dimentions of Matrix: "))
    iters = int(input("Number of Iterations to attempt: "))
    posDefQ = 0
    posDefLambda = 0

    #Generates INV samples
    for i in range(iters):

        # Generate random square matrix
        n = random.randint(2, maxdim)
        df = random.randrange(n, pow(n, n))

        while True:
            # create PSD Matarix
            # https://math.stackexchange.com/questions/332456/how-to-make-a-matrix-positive-semidefinite
            random_matrix = np.random.rand(n, n)
            PSD_Random = np.dot(random_matrix, random_matrix.transpose())
            #check to see if PD, if so, continnue
            if is_pos_def(PSD_Random):
                break

        #PSD_random is Gaurenteed PD at this point

        #Using default size=1, random_state=None
        iw = invwishart.rvs(df=df, scale=PSD_Random)
        iw = np.linalg.inv(iw)
        #calculate row mean from cov matrix
        iw_mean = np.mean(iw, axis=0)
        Q, Lambda_zz = vals_from_priors(iw, iw_mean, math.floor(iw_mean.shape[0] * .25))

        if is_pos_def(Q):
            posDefQ = posDefQ + 1
        if is_pos_def(Lambda_zz):
            posDefLambda = posDefLambda + 1

    print(f"{posDefQ}/{iters} Qs are PD")
    print(f"{posDefQ}/{iters} Lambda_zz are PD")


