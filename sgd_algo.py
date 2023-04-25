from math import sqrt
import numpy as np

def check_gd_bounds(constraint, curr_pos, step):
    new_pos = curr_pos
    for i in range(len(curr_pos)):
        new_pos[i] = curr_pos[i] + step[i]
        if new_pos[i] > constraint[0][i]:
                new_pos[i] = constraint[0][i]
        if new_pos[i] < constraint[1][i]:
            new_pos[i] = constraint[1][i]
    print(new_pos)
    return new_pos

def adaGrad(gradient, constraint, curr_pos, R, learn_rate=0.001, epsilon = 0.0001):
    direction = gradient(curr_pos) / np.linalg.norm(gradient(curr_pos))
    R_t_1 = R + np.outer(direction,direction)
    diag= np.diagonal(R_t_1) #Gives sqrt of each element of the diag elementwise
    step = learn_rate * (np.sqrt(diag + epsilon))**(-1) * direction
    return check_gd_bounds(constraint, curr_pos, step), R_t_1

def RMSProp(gradient, constraint, curr_pos, R, learn_rate= 0.001, beta = .9, epsilon = .00001):
    direction = gradient(curr_pos) / np.linalg.norm(gradient(curr_pos))
    R_plus = R + np.outer(direction, direction)
    R_t_1 = (beta * R) + ((1-beta) * R_plus)
    diag = np.diagonal(R_t_1)  # Gives sqrt of each element of the diag elementwise
    step = learn_rate * (np.sqrt(diag + epsilon)) ** (-1) * direction
    return check_gd_bounds(constraint, curr_pos, step), R_t_1

def adam(gradient, constraint, curr_pos, R, learn_rate= 0.001, t = 0, m = 0, beta_1 = .9, beta_2 = .999, epsilon = 0.00001):
    direction = gradient(curr_pos) / np.linalg.norm(gradient(curr_pos))
    m_t_1 = (beta_1 * m) + ((1 - beta_1) * gradient(curr_pos))
    m_t_1_hat = m_t_1 / (1 - pow(beta_1, t))
    R_plus = R + np.outer(direction, direction)
    R_t_1 = (beta_2 * R) + ((1-beta_2) * R_plus)
    R_t_1_hat = R_t_1 / (1 - pow(beta_2, t))
    diag = np.diagonal(R_t_1_hat)
    step = learn_rate * (np.sqrt(diag + epsilon)) ** (-1) * m_t_1_hat
    #v_t_1 = (beta_1 * v) + ((1 - beta_2) * pow(np.linalg.norm(gradient(curr_pos)), 2))
    #v_t_1_hat = v_t_1 / (1 - pow(beta_2, t))
    #step = (learn_rate / (np.sqrt(v_t_1_hat) + epsilon)) * m_t_1_hat
    return check_gd_bounds(constraint, curr_pos, step), R_t_1, m_t_1 #
