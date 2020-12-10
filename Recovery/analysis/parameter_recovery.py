from functions import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import scipy.stats as sc
import matplotlib.patches as mpatches

print("------------------------------------")
print("Parameter recovery")

cpt_parameter_recovery_file_1 = 'parameter_recovery_CPT_chunk_1.mat'
cpt_parameter_recovery_file_2 = 'parameter_recovery_CPT_chunk_2.mat'
cpt_parameter_recovery_file_3 = 'parameter_recovery_CPT_chunk_3.mat'
cpt_ground_truth_file         = "Choices_simulated_from_CPT.mat"

LML_parameter_recovery_file_1 = 'parameter_recovery_LML_chunk_1.mat'
LML_parameter_recovery_file_2 = 'parameter_recovery_LML_chunk_2.mat'
LML_parameter_recovery_file_3 = 'parameter_recovery_LML_chunk_3.mat'

x = np.linspace(0,1,300)

show_cpt = True
show_lml = True

A      = [0,1,2,3]
marker = ['^','s','o','v']
color  = ['r','b','g','m']

if show_cpt:
    print("------------------------------------")
    print("Evaluating CPT data...\n")
    print("Reading output...")

    _,beta_cpt_1,   delta_cpt_1,   gamma_cpt_1    = read_output(cpt_parameter_recovery_file_1,'parameter_recovery')
    _,beta_cpt_2,   delta_cpt_2,   gamma_cpt_2    = read_output(cpt_parameter_recovery_file_2,'parameter_recovery')
    _,beta_cpt_3,   delta_cpt_3,   gamma_cpt_3    = read_output(cpt_parameter_recovery_file_3,'parameter_recovery')
    _,beta_cpt_true,delta_cpt_true,gamma_cpt_true = read_output(cpt_ground_truth_file,        'parameter_recovery')

    n_agents = np.shape(beta_cpt_1)[0]
    n_samples = np.shape(beta_cpt_1)[1]
    n_chains = np.shape(beta_cpt_1)[2]

    print('---')
    print(f"Number of agents = {n_agents}")
    print(f"Number of chains = {n_chains}")
    print(f"Number of samples = {n_samples}")
    print('---')

    #--------delta----------#
    print("\nProcessing Delta...")
    _,map_agent_cpt_delta_1 = process_params(delta_cpt_1, n_agents, n_chains, n_samples, output="map")
    _,map_agent_cpt_delta_2 = process_params(delta_cpt_2, n_agents, n_chains, n_samples, output="map")
    _,map_agent_cpt_delta_3 = process_params(delta_cpt_3, n_agents, n_chains, n_samples, output="map")

    corr1,_ = sc.pearsonr(map_agent_cpt_delta_1,delta_cpt_true[:,0,0])
    corr2,_ = sc.pearsonr(map_agent_cpt_delta_2,delta_cpt_true[:,0,0])
    corr3,_ = sc.pearsonr(map_agent_cpt_delta_3,delta_cpt_true[:,0,0])
    print(f"Pearson correlation coefficient for Delta in chunk 1: {corr1:.3f}")
    print(f"Pearson correlation coefficient for Delta in chunk 2: {corr2:.3f}")
    print(f"Pearson correlation coefficient for Delta in chunk 3: {corr3:.3f}")

    plt.figure()
    plt.suptitle("True delta vs estimated delta",fontsize=18)
    for i in range(n_agents):
        # plt.subplot(1,2,1)
        if i in A:
            plt.scatter(delta_cpt_true[i,0,0],map_agent_cpt_delta_1[i], marker=marker[i], edgecolor='b', facecolors='none')
            plt.scatter(delta_cpt_true[i,0,0],map_agent_cpt_delta_2[i], marker=marker[i], edgecolor='r', facecolors='none')
            plt.scatter(delta_cpt_true[i,0,0],map_agent_cpt_delta_3[i], marker=marker[i], edgecolor='g', facecolors='none')
        else:
            plt.scatter(delta_cpt_true[i,0,0],map_agent_cpt_delta_1[i], marker='x', c='b')
            plt.scatter(delta_cpt_true[i,0,0],map_agent_cpt_delta_2[i], marker='x', c='r')
            plt.scatter(delta_cpt_true[i,0,0],map_agent_cpt_delta_3[i], marker='x', c='g')
        plt.ylim([0,1.8])
        plt.xlim([0,1.8])
    #Dummies for legend
    plt.scatter(10,10,c='b',label="Chunk 1", marker=">")
    plt.scatter(10,10,c='r',label="Chunk 2", marker=">")
    plt.scatter(10,10,c='g',label="Chunk 3", marker=">")
    plt.scatter(10,10, facecolors='none',label=" ")
    plt.scatter(10,10,label="Agent 1", marker=marker[0], edgecolor='k', facecolors='none')
    plt.scatter(10,10,label="Agent 2", marker=marker[1], edgecolor='k', facecolors='none')
    plt.scatter(10,10,label="Agent 3", marker=marker[2], edgecolor='k', facecolors='none')
    plt.scatter(10,10,label="Agent 4", marker=marker[3], edgecolor='k', facecolors='none')
    plt.scatter(10,10,marker='x',c='k',label="Agents 5-10")
    plt.xlabel("$\\delta_{'Ground-truth'}$",fontsize=14)
    plt.ylabel("$\delta_e$",fontsize=14)
    plt.legend(loc='upper left')

    #--------gamma----------#
    print("\nProcessing Gamma...")
    _,map_agent_cpt_gamma_1 = process_params(gamma_cpt_1, n_agents, n_chains, n_samples, output="map")
    _,map_agent_cpt_gamma_2 = process_params(gamma_cpt_2, n_agents, n_chains, n_samples, output="map")
    _,map_agent_cpt_gamma_3 = process_params(gamma_cpt_3, n_agents, n_chains, n_samples, output="map")

    corr1,_ = sc.pearsonr(map_agent_cpt_gamma_1,gamma_cpt_true[:,0,0])
    corr2,_ = sc.pearsonr(map_agent_cpt_gamma_2,gamma_cpt_true[:,0,0])
    corr3,_ = sc.pearsonr(map_agent_cpt_gamma_3,gamma_cpt_true[:,0,0])
    print(f"Pearson correlation coefficient for Gamma in chunk 1: {corr1:.3f}")
    print(f"Pearson correlation coefficient for Gamma in chunk 2: {corr2:.3f}")
    print(f"Pearson correlation coefficient for Gamma in chunk 3: {corr3:.3f}")

    plt.figure()
    plt.suptitle("True gamma vs estimated gamma",fontsize=18)
    for i in range(n_agents):
        # plt.subplot(1,2,1)
        if i in A:
            plt.scatter(gamma_cpt_true[i,0,0],map_agent_cpt_gamma_1[i], marker=marker[i], edgecolor='k', facecolors='none')
            plt.scatter(gamma_cpt_true[i,0,0],map_agent_cpt_gamma_2[i], marker=marker[i], edgecolor='r', facecolors='none')
            plt.scatter(gamma_cpt_true[i,0,0],map_agent_cpt_gamma_3[i], marker=marker[i], edgecolor='g', facecolors='none')
        else:
            plt.scatter(gamma_cpt_true[i,0,0],map_agent_cpt_gamma_1[i], marker='x', c='k')
            plt.scatter(gamma_cpt_true[i,0,0],map_agent_cpt_gamma_2[i], marker='x', c='r')
            plt.scatter(gamma_cpt_true[i,0,0],map_agent_cpt_gamma_3[i], marker='x', c='g')
        plt.ylim([0,1.8])
        plt.xlim([0,1.8])
    #Dummies for legend
    plt.scatter(10,10,c='k',label="Chunk 1", marker=">")
    plt.scatter(10,10,c='r',label="Chunk 2", marker=">")
    plt.scatter(10,10,c='g',label="Chunk 3", marker=">")
    plt.scatter(10,10, facecolors='none',label=" ")
    plt.scatter(10,10,label="Agent 1", marker=marker[0], edgecolor='k', facecolors='none')
    plt.scatter(10,10,label="Agent 2", marker=marker[1], edgecolor='k', facecolors='none')
    plt.scatter(10,10,label="Agent 3", marker=marker[2], edgecolor='k', facecolors='none')
    plt.scatter(10,10,label="Agent 4", marker=marker[3], edgecolor='k', facecolors='none')
    plt.scatter(10,10,marker='x',c='k',label="Agents 5-10")
    plt.xlabel("$\\gamma_{'Ground-truth'}$",fontsize=14)
    plt.ylabel("$\gamma_e$",fontsize=14)
    plt.legend(loc='upper left')

    # #--------PW----------#
    print("\nProcessing w...")
    plt.figure()
    plt.suptitle("Estimated delta vs estimated gamma",fontsize=18)
    for i in range(n_agents):
        # plt.subplot(1,2,1)
        if i in A:
            plt.scatter(map_agent_cpt_delta_1[i],map_agent_cpt_gamma_1[i], marker=marker[i], edgecolor='b', facecolors='none')
            plt.scatter(map_agent_cpt_delta_2[i],map_agent_cpt_gamma_2[i], marker=marker[i], edgecolor='r', facecolors='none')
            plt.scatter(map_agent_cpt_delta_3[i],map_agent_cpt_gamma_3[i], marker=marker[i], edgecolor='g', facecolors='none')
        else:
            plt.scatter(map_agent_cpt_delta_1[i],map_agent_cpt_gamma_1[i], marker='x', c='b')
            plt.scatter(map_agent_cpt_delta_2[i],map_agent_cpt_gamma_2[i], marker='x', c='r')
            plt.scatter(map_agent_cpt_delta_3[i],map_agent_cpt_gamma_3[i], marker='x', c='g')
        plt.ylim([0,1.8])
        plt.xlim([0,1.8])
    #Dummies for legend
    plt.scatter(10,10,c='b',label="Chunk 1", marker=">")
    plt.scatter(10,10,c='r',label="Chunk 2", marker=">")
    plt.scatter(10,10,c='g',label="Chunk 3", marker=">")
    plt.scatter(10,10, facecolors='none',label=" ")
    plt.scatter(10,10,label="Agent 1", marker=marker[0], edgecolor='k', facecolors='none')
    plt.scatter(10,10,label="Agent 2", marker=marker[1], edgecolor='k', facecolors='none')
    plt.scatter(10,10,label="Agent 3", marker=marker[2], edgecolor='k', facecolors='none')
    plt.scatter(10,10,label="Agent 4", marker=marker[3], edgecolor='k', facecolors='none')
    plt.scatter(10,10,marker='x',c='k',label="Agents 5-10")
    plt.xlabel("$\\delta_e$",fontsize=14)
    plt.ylabel("$\gamma_e$",fontsize=14)
    plt.legend(loc='upper left')
  
    
    

    for i in A: #agents
        plt.figure()
        plt.suptitle(f"Weighting function envolvment for CPT-Agent {i+1}", fontsize=18)
        
        plt.subplot(2,3,1)
        plt.title(f"Chunk 1",fontsize=16)
        plt.scatter(map_agent_cpt_delta_1[i],map_agent_cpt_gamma_1[i], edgecolor='b', facecolors='none', marker='^', label="Estimated")
        plt.scatter(delta_cpt_true[i,0,0],gamma_cpt_true[i,0,0], edgecolor='r',facecolor='none', label="'Ground truth'")
        # plt.scatter(1,1,c='k', marker='x', label="No weighting")
        plt.ylim([0,1.8])
        plt.xlim([0,1.8])
        plt.xlabel("$\\delta$",fontsize=14)
        plt.ylabel("$\\gamma$",fontsize=14)
        plt.legend(loc='upper left')
        plt.subplot(2,3,2)
        plt.title(f"Chunk 2",fontsize=16)
        plt.scatter(map_agent_cpt_delta_2[i],map_agent_cpt_gamma_2[i], edgecolor='b',facecolors='none', marker='^', label="Estimated")
        plt.scatter(delta_cpt_true[i,0,0],gamma_cpt_true[i,0,0], edgecolor='r',facecolor='none', marker='o', label="'Ground truth'")
        # plt.scatter(1,1,c='k',marker='x', label="No weighting")
        plt.ylim([0,1.8])
        plt.xlim([0,1.8])
        plt.xlabel("$\\delta$",fontsize=14)
        plt.ylabel("$\\gamma$",fontsize=14)
        plt.legend(loc='upper left')
        plt.subplot(2,3,3)
        plt.title(f"Chunk 3",fontsize=16)
        plt.scatter(map_agent_cpt_delta_3[i],map_agent_cpt_gamma_3[i], edgecolor='b',facecolors='none', marker='^', label="Estimated")
        plt.scatter(delta_cpt_true[i,0,0],gamma_cpt_true[i,0,0], edgecolor='r',facecolor='none',marker='o', label="'Ground truth'")
        # plt.scatter(1,1,c='k',marker='x', label="No weighting")
        plt.ylim([0,1.8])
        plt.xlim([0,1.8])
        plt.xlabel("$\\delta$",fontsize=14)
        plt.ylabel("$\gamma$",fontsize=14)
        plt.legend(loc='upper left')
        
        w_1 = cpt_weighting_function(x, map_agent_cpt_delta_1[i],map_agent_cpt_gamma_1[i])
        w_2 = cpt_weighting_function(x, map_agent_cpt_delta_2[i],map_agent_cpt_gamma_2[i])
        w_3 = cpt_weighting_function(x, map_agent_cpt_delta_3[i],map_agent_cpt_gamma_3[i])
        w_true = cpt_weighting_function(x, delta_cpt_true[i,0,0], gamma_cpt_true[i,0,0])

        plt.subplot(2,3,4)
        plt.plot(x,w_1, 'b--', label="Estimated")
        plt.plot(x,w_true, 'r-.', label="'Ground truth'")
        plt.plot(x,x,'k-', label="No weighting")
        plt.xlabel("$\hat{p}(x)$",fontsize=14)
        plt.ylabel("$w(x)$",fontsize=14)
        plt.legend(loc='upper left')
        plt.subplot(2,3,5)
        plt.plot(x,w_2, 'b--', label="Estimated")
        plt.plot(x,w_true, 'r-.', label="'Ground truth'")
        plt.plot(x,x,'k-', label="No weighting")
        plt.xlabel("$\hat{p}(x)$",fontsize=14)
        plt.ylabel("$w(x)$",fontsize=14)
        plt.legend(loc='upper left')
        plt.subplot(2,3,6)
        plt.plot(x,w_3, 'b--', label="Estimated")
        plt.plot(x,w_true, 'r-.', label="'Ground truth'")
        plt.plot(x,x,'k-', label="No weighting")
        plt.xlabel("$\hat{p}(x)$",fontsize=14)
        plt.ylabel("$w(x)$",fontsize=14)
        plt.legend(loc='upper left')

    print("\nPlotting...")

if show_lml:
    print("------------------------------------")
    print("Evaluating LML data...\n")
    print("Reading output...")

    _,beta_lml_1,   delta_lml_1,   gamma_lml_1    = read_output(LML_parameter_recovery_file_1,'parameter_recovery')
    _,beta_lml_2,   delta_lml_2,   gamma_lml_2    = read_output(LML_parameter_recovery_file_2,'parameter_recovery')
    _,beta_lml_3,   delta_lml_3,   gamma_lml_3    = read_output(LML_parameter_recovery_file_3,'parameter_recovery')

    n_agents = np.shape(beta_lml_1)[0]
    n_samples = np.shape(beta_lml_1)[1]
    n_chains = np.shape(beta_lml_1)[2]

    print('---')
    print(f"Number of agents = {n_agents}")
    print(f"Number of chains = {n_chains}")
    print(f"Number of samples = {n_samples}")
    print('---')

    #--------delta----------#
    print("\nProcessing Delta...")
    _,map_agent_lml_delta_1 = process_params(delta_lml_1, n_agents, n_chains, n_samples, output="map")
    _,map_agent_lml_delta_2 = process_params(delta_lml_2, n_agents, n_chains, n_samples, output="map")
    _,map_agent_lml_delta_3 = process_params(delta_lml_3, n_agents, n_chains, n_samples, output="map")

    #--------gamma----------#
    print("\nProcessing Gamma...")
    _,map_agent_lml_gamma_1 = process_params(gamma_lml_1, n_agents, n_chains, n_samples, output="map")
    _,map_agent_lml_gamma_2 = process_params(gamma_lml_2, n_agents, n_chains, n_samples, output="map")
    _,map_agent_lml_gamma_3 = process_params(gamma_lml_3, n_agents, n_chains, n_samples, output="map")

    #--------PW----------#
    print("\nProcessing w...")
    plt.figure()
    plt.suptitle("Estimated delta vs estimated gamma", fontsize=18)
    for i in range(n_agents):
        if i in A:
            plt.scatter(map_agent_lml_delta_1[i],map_agent_lml_gamma_1[i], marker=marker[i], edgecolor='b', facecolors='none')
            plt.scatter(map_agent_lml_delta_2[i],map_agent_lml_gamma_2[i], marker=marker[i], edgecolor='r', facecolors='none')
            plt.scatter(map_agent_lml_delta_3[i],map_agent_lml_gamma_3[i], marker=marker[i], edgecolor='g', facecolors='none')
        else:
            plt.scatter(map_agent_lml_delta_1[i],map_agent_lml_gamma_1[i], marker='x', c='b')
            plt.scatter(map_agent_lml_delta_2[i],map_agent_lml_gamma_2[i], marker='x', c='r')
            plt.scatter(map_agent_lml_delta_3[i],map_agent_lml_gamma_3[i], marker='x', c='g')
        plt.ylim([0.7,1.05])
        plt.xlim([0.9,1.05])
    plt.xlabel("$\\delta$",fontsize=14)
    plt.ylabel("$\gamma$",fontsize=14)
    plt.scatter(10,10,c='b',label="Chunk 1", marker=">")
    plt.scatter(10,10,c='r',label="Chunk 2", marker=">")
    plt.scatter(10,10,c='g',label="Chunk 3", marker=">")
    plt.scatter(10,10, facecolors='none',label=" ")
    plt.scatter(10,10, marker=marker[0], edgecolor='k', facecolors='none', label=f"Agent 1")
    plt.scatter(10,10, marker=marker[1], edgecolor='k', facecolors='none', label=f"Agent 2")
    plt.scatter(10,10, marker=marker[2], edgecolor='k', facecolors='none', label=f"Agent 3")
    plt.scatter(10,10, marker=marker[3], edgecolor='k', facecolors='none', label=f"Agent 4")
    plt.scatter(10,10,marker='x',c='k',label="Agents 5-10")
    plt.legend(loc='upper left')

    for i in range(1): #agents
        plt.figure()
        plt.suptitle(f"Weighting function envolvment for CPT-Agent {i+1}", fontsize=18)
        plt.subplot(2,3,1)
        plt.title(f"Chunk 1",fontsize=16)
        plt.scatter(map_agent_lml_delta_1[i],map_agent_lml_gamma_1[i], edgecolor='b', facecolors='none', marker='^', label="Estimated")
        # plt.scatter(1,1,c='k', marker='x', label="No weighting")
        plt.ylim([0.7,1.05])
        plt.xlim([0.9,1.05])
        plt.xlabel("$\\delta$",fontsize=14)
        plt.ylabel("$\\gamma$",fontsize=14)
        plt.legend(loc='upper left')
        plt.subplot(2,3,2)
        plt.title(f"Chunk 2",fontsize=16)
        plt.scatter(map_agent_lml_delta_2[i],map_agent_lml_gamma_2[i], edgecolor='b',facecolors='none', marker='^', label="Estimated")
        # plt.scatter(1,1,c='k',marker='x', label="No weighting")
        plt.ylim([0.7,1.05])
        plt.xlim([0.9,1.05])
        plt.xlabel("$\\delta$",fontsize=14)
        plt.ylabel("$\\gamma$",fontsize=14)
        plt.legend(loc='upper left')
        plt.subplot(2,3,3)
        plt.title(f"Chunk 3",fontsize=16)
        plt.scatter(map_agent_lml_delta_3[i],map_agent_lml_gamma_3[i], edgecolor='b',facecolors='none', marker='^', label="Estimated")
        # plt.scatter(1,1,c='k',marker='x', label="No weighting")
        plt.ylim([0.7,1.05])
        plt.xlim([0.9,1.05])
        plt.xlabel("$\\delta$",fontsize=14)
        plt.ylabel("$\gamma$",fontsize=14)
        plt.legend(loc='upper left')

        w_1 = cpt_weighting_function(x, map_agent_lml_delta_1[i],map_agent_lml_gamma_1[i])
        w_2 = cpt_weighting_function(x, map_agent_lml_delta_2[i],map_agent_lml_gamma_2[i])
        w_3 = cpt_weighting_function(x, map_agent_lml_delta_3[i],map_agent_lml_gamma_3[i])
        w_true_1 = lml_weighting_function(x,10)
        w_true_2 = lml_weighting_function(x,50)
        w_true_3 = lml_weighting_function(x,100)

        plt.subplot(2,3,4)
        plt.plot(x,w_1, 'b--', label="Estimated")
        plt.plot(x,w_true_1,'r-.', label="Predicted")
        plt.plot(x,x,'k-', label = "No weighting")
        plt.xlabel("$\hat{p}(x)$", fontsize=14)
        plt.ylabel("$w(x)$", fontsize=14)
        plt.legend(loc='upper left')
        plt.subplot(2,3,5)
        plt.plot(x,w_2,'b--', label="Estimated")
        plt.plot(x,w_true_2,'r-.', label="Predicted")
        plt.plot(x,x,'k-', label = "No weighting")
        plt.xlabel("$\hat{p}(x)$", fontsize=14)
        plt.ylabel("$w(x)$", fontsize=14)
        plt.legend(loc='upper left')
        plt.subplot(2,3,6)
        plt.plot(x,w_3,'b--', label="Estimated")
        plt.plot(x,w_true_3,'r-.', label="Predicted")
        plt.plot(x,x,'k-', label = "No weighting")
        plt.xlabel("$\hat{p}(x)$", fontsize=14)
        plt.ylabel("$w(x)$", fontsize=14)
        plt.legend(loc='upper left')
    
    print("\nPlotting...")
plt.show()

print("------------------------------------")