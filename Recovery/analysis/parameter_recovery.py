from functions import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import scipy.stats as sc

cpt_file_name = 'parameter_recovery_CPT_Gamble_2.mat'
lml_file_name = "parameter_recovery_LML_Gamble_2.mat"

print("------------------------------------")
print("Evaluating CPT data...\n")
print("Reading output...")
_,beta,delta,gamma = read_output(cpt_file_name,'parameter_recovery')
_,beta_true,delta_true,gamma_true = read_output('Choices_simulated_from_CPT_Gamble_2.mat','parameter_recovery')

n_chunks = np.shape(beta)[0]
n_agents = np.shape(beta)[1]
n_samples = np.shape(beta)[2]
n_chains = np.shape(beta)[3]

print('---')
print(f"Number of agents = {n_agents}")
print(f"Number of chunks = {n_chunks}")
print(f"Number of chains = {n_chains}")
print(f"Number of samples = {n_samples}")
print('---')

#--------delta----------#
print("\nProcessing Delta...")
_,_, map_agent_delta = process_params(delta, n_chunks, n_agents, n_chains, n_samples, output="map")

plt.figure()
plt.suptitle("True value vs estimated (Delta) for each agent")
for c in range(n_chunks):
    plt.subplot(1,5,c+1)
    plt.title(f"Chunk {c+1}")
    plt.xlabel("$\\delta_{est}$")
    plt.ylabel("$\\delta_{true}$")
    plt.scatter(map_agent_delta[c], delta_true[:,0,0])
    plt.xlim([0,2])
    plt.ylim([0,2])
    corr,_ = sc.pearsonr(map_agent_delta[c],delta_true[:,0,0])
    print(f"Pearson correlation coefficient for Delta in chunk {c+1}: {corr}")

#--------gamma----------#
print("\nProcessing Gamma...")

_,_, map_agent_gamma = process_params(gamma, n_chunks, n_agents, n_chains, n_samples, output="map")

for c in range(n_chunks):
    plt.figure()
    plt.scatter(map_agent_gamma[c], gamma_true[:,0,0])
    plt.xlim([0,2])
    plt.ylim([0,2])
    corr,_ = sc.pearsonr(map_agent_gamma[c],gamma_true[:,0,0])
    print("Pearson correlation coefficient for Gamma: ", corr)


#--------PW----------#
print("\nProcessing w...")
for c in range(n_chunks):
    plt.figure()
    plt.scatter(map_agent_gamma[c], map_agent_delta[c])
    plt.xlim([0,2])
    plt.ylim([0,2])

plt.figure()
plt.suptitle("Weighting function envolvment for CPT-species", fontsize=18)
x = np.linspace(0,1,100)
for c in range(n_chunks):
    w = []
    w_true = []
    plt.subplot(1,5,c+1)
    for i in range(1):
        w.append(cpt_weighting_function(x,map_agent_gamma[c][i], map_agent_delta[c][i]))
        w_true.append(cpt_weighting_function(x, gamma_true[i,0,0], delta_true[i,0,0]))
    
        
        plt.plot(x,w[i], label="Estimated")
        plt.plot(x,w_true[i], label="True")
        plt.ylabel("Probability weight - $w(x)$", fontsize=14)
        plt.xlabel("Probability - $p(x)$", fontsize=14)
        plt.legend()

print("\nPlotting...")
plt.show()


plt.show()



print("------------------------------------")
print("Evaluating LML data...\n")
print("Reading output...")
_,beta_LML,delta_LML,gamma_LML = read_output(lml_file_name,'parameter_recovery')

n_chunks = np.shape(delta_LML)[0]
n_agents = np.shape(delta_LML)[1]
n_samples = np.shape(delta_LML)[2]
n_chains = np.shape(delta_LML)[3]

print("\nProcessing Delta...")
_,_, map_agent_delta_LML = process_params(delta_LML, n_chunks, n_agents, n_chains, n_samples, output="map")
print("\nProcessing Gamma...")
_,_, map_agent_gamma_LML = process_params(gamma_LML, n_chunks, n_agents, n_chains, n_samples, output="map")

#--------PW----------#
print("\nProcessing w...")
plt.figure()
for c in range(n_chunks):
    plt.subplot(1,5,c+1)
    plt.scatter(map_agent_gamma_LML[c], map_agent_delta_LML[c])
    plt.xlim([0,2])
    plt.ylim([0,2])

plt.figure()
plt.suptitle("Weighting function envolvment for CPT-species", fontsize=18)
x = np.linspace(0,1,100)
for c in range(n_chunks):
    w = []
    w_true = []
    plt.subplot(1,5,c+1)
    for i in range(1):
        w.append(cpt_weighting_function(x,map_agent_gamma_LML[c][i], map_agent_delta_LML[c][i])) 
        
        plt.plot(x,w[i], label="Estimated")
        plt.ylabel("Probability weight - $w(x)$", fontsize=14)
        plt.xlabel("Probability - $p(x)$", fontsize=14)
        plt.legend()

print("\nPlotting...")
plt.show()