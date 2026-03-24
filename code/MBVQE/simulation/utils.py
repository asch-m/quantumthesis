import numpy as np

def true_ground_state_energy_and_state(ham):
    eig, eigv = np.linalg.eigh(ham)
    return eig[0], eigv[:,0]

def get_exp_val_state_vec(state_vec, op):
    return np.dot(np.conjugate(state_vec),np.dot(op, state_vec))

def get_exp_val_density(dense_mat, op):
    return np.trace(np.dot(op, dense_mat))