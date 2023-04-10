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

    return new_pos

def adaGrad(gradient, constraint, curr_pos, learn_rate=0.001, v = 0):
    step = (learn_rate / sqrt(v + (.00001))) * gradient(curr_pos)
    v_t_1 = v + pow(np.linalg.norm(gradient(curr_pos)), 2)
    return check_gd_bounds(constraint, curr_pos, step), v_t_1

def RMSProp(gradient, constraint, curr_pos, learn_rate= 0.001, v = 0, beta = .9):
    step = ((learn_rate / sqrt(v + (.00001))) * gradient(curr_pos))
    v_t_1 = (beta * v) + ((1-beta) * pow(np.linalg.norm(gradient(curr_pos)), 2))
    return check_gd_bounds(constraint, curr_pos, step), v_t_1

def adam(gradient, constraint, curr_pos, learn_rate= 0.001, t = 0, v = 0, m = 0, beta_1 = .9, beta_2 = .999):
    m_t_1 = (beta_1 * m) + ((1 - beta_1) * gradient(curr_pos))
    v_t_1 = (beta_1 * v) + ((1 - beta_2) * pow(np.linalg.norm(gradient(curr_pos)), 2))
    m_t_1_hat = m_t_1 / (1 - pow(beta_1, t))
    v_t_1_hat = v_t_1 / (1 - pow(beta_2, t))
    step = (learn_rate / (np.sqrt(v_t_1_hat) + .00001)) * m_t_1_hat
    return check_gd_bounds(constraint, curr_pos, step), v_t_1, m_t_1