from functions import read_output, process_params, cpt_weighting_function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

cpt_file_name = 'parameter_recovery_CPT_Gamble_1.mat'
lml_file_name = "test"

print("------------------------------------")
print("Evaluating CPT data...\n")
print("Reading output...")
_,_,beta_cpt,beta_lml,delta,gamma = read_output(cpt_file_name,'parameter_recovery')
_,_,beta_cpt_true,beta_lml_true,delta_true,gamma_true = read_output('Choices_simulated_from_CPT_Gamble_1.mat','parameter_recovery')

n_chunks = np.shape(beta_cpt)[0]
n_subjects = np.shape(beta_cpt)[1]
n_samples = np.shape(beta_cpt)[2]
n_chains = np.shape(beta_cpt)[3]

print('---')
print(f"Number of subjects = {n_subjects}")
print(f"Number of chunks = {n_chunks}")
print(f"Number of chains = {n_chains}")
print(f"Number of samples = {n_samples}")
print('---')

#--------delta----------#
print("\nProcessing Delta...")
map_marginal_delta, map_chunks_delta, map_subjects_delta = process_params(delta, n_chunks, n_subjects, n_chains, n_samples, output="map")

plt.figure()
for i in range(n_chunks):
    plt.subplot(2,5,i+1)
    plt.scatter(delta_true,map_subjects_delta[i])

#--------gamma----------#
print("\nProcessing Gamma...")
map_marginal_gamma, map_chunks_gamma, map_subjects_gamma = process_params(gamma, n_chunks, n_subjects, n_chains, n_samples, output="map")

plt.figure()
for i in range(n_chunks):
    plt.subplot(2,5,i+1)
    plt.scatter(gamma_true,map_subjects_gamma[i])


#--------w over chunks (delta,gamma)----------#
print("\nProcessing w...")
plt.figure()
for i in range(n_chunks):
    plt.suptitle("Weighting function parameters envolvment over time for CPT species")
    plt.subplot(2,5,i+1)
    plt.title(f"Chunk {i+1}")
    plt.scatter(map_chunks_delta[i], map_chunks_gamma[i])
    plt.xlim([0,1])
    plt.ylim([0,1])

plt.figure()
x = np.linspace(0,1,100)
for i in range(n_chunks):
    w = cpt_weighting_function(x,map_chunks_gamma[i], map_chunks_delta[i])
    w_true = cpt_weighting_function(x, np.median(gamma_true), np.median(delta_true))
    
    plt.subplot(2,5,i+1)
    plt.plot(x,w, label="est")
    plt.plot(x,w_true, label="True")
    plt.legend()

print("\nPlotting...")
plt.show()

print("------------------------------------")
print("Evaluating LML data\n")
print("Reading output...")
_,_,beta_cpt,beta_lml,delta,gamma = read_output(lml_file_name,'parameter_recovery')
n_chunks = np.shape(beta_cpt)[0]
n_subjects = np.shape(beta_cpt)[1]
n_samples = np.shape(beta_cpt)[2]
n_chains = np.shape(beta_cpt)[3]

print('---')
print(f"Number of subjects = {n_subjects}")
print(f"Number of chunks = {n_chunks}")
print(f"Number of chains = {n_chains}")
print(f"Number of samples = {n_samples}")
print('---')

#--------delta----------#
print("\nProcessing Delta...")
map_marginal_delta, map_chunks_delta, map_subjects_delta = process_params(delta, n_chunks, n_subjects, n_chains, n_samples, output="map")

#--------gamma----------#
print("\nProcessing Gamma...")
map_marginal_gamma, map_chunks_gamma, map_subjects_gamma = process_params(gamma, n_chunks, n_subjects, n_chains, n_samples, output="map")

#--------w over chunks (delta,gamma)----------#
print("\nProcessing w...")
plt.figure()
for i in range(n_chunks):
    plt.suptitle("Weighting function parameters envolvment over time for CPT species")
    plt.subplot(2,5,i+1)
    plt.title(f"Chunk {i+1}")
    plt.scatter(map_chunks_delta[i], map_chunks_gamma[i])
    plt.xlim([0,1])
    plt.ylim([0,1])

plt.figure()
x = np.linspace(0,1,100)
for i in range(n_chunks):
    w = cpt_weighting_function(x,map_chunks_gamma[i], map_chunks_delta[i])
    
    plt.subplot(2,5,i+1)
    plt.plot(x,w, label="est")
    plt.plot(x,w_true, label="True")
    plt.legend()

print("\nPlotting...")
plt.show()