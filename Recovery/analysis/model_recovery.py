from Read_output import read_output
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

_,_,beta_cpt,beta_lml,delta,gamma = read_output('parameter_recovery_CPT_Gamble_1.mat','parameter_recovery')
_,_,beta_cpt_true,beta_lml_true,delta_true,gamma_true = read_output('Choices_simulated_from_CPT_Gamble_1.mat','parameter_recovery')

n_subjects = np.shape(beta_cpt)[0]
n_samples = np.shape(beta_cpt)[1]
n_trials = np.shape(beta_cpt)[2]

print('---------------------------------------')
print(f"Number of subjects = {n_subjects}")
print(f"Number of trials = {n_trials}")
print(f"Number of samples = {n_samples}")
print('---------------------------------------')
