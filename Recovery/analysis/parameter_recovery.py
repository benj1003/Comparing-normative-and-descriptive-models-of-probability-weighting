from functions import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import scipy.stats as sc

cpt_parameter_recovery_file_1 = 'parameter_recovery_CPT_chunk_1.mat'
cpt_parameter_recovery_file_2 = 'parameter_recovery_CPT_chunk_2.mat'
cpt_parameter_recovery_file_3 = 'parameter_recovery_CPT_chunk_3.mat'
cpt_ground_truth_file         = "Choices_simulated_from_CPT.mat"

LML_parameter_recovery_file_1 = 'parameter_recovery_LML_chunk_1.mat'
LML_parameter_recovery_file_2 = 'parameter_recovery_LML_chunk_2.mat'
LML_parameter_recovery_file_3 = 'parameter_recovery_LML_chunk_3.mat'

x = np.linspace(0,1,300)

show_cpt = False
show_lml = True

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

    plt.figure()
    plt.title("True vs. estimated Delta (CPT)")
    plt.subplot(1,3,1)
    plt.scatter(delta_cpt_true[:,0,0],map_agent_cpt_delta_1)
    plt.plot(x,x)
    plt.ylabel("Estimated")
    plt.xlabel("True")
    plt.ylim([0,3])
    plt.xlim([0,3])
    plt.subplot(1,3,2)
    plt.scatter(delta_cpt_true[:,0,0],map_agent_cpt_delta_2)
    plt.ylabel("Estimated")
    plt.xlabel("True")
    plt.ylim([0,3])
    plt.xlim([0,3])
    plt.subplot(1,3,3)
    plt.scatter(delta_cpt_true[:,0,0],map_agent_cpt_delta_3)
    plt.ylabel("Estimated")
    plt.xlabel("True")
    plt.ylim([0,3])
    plt.xlim([0,3])

    #--------gamma----------#
    print("\nProcessing Gamma...")
    _,map_agent_cpt_gamma_1 = process_params(gamma_cpt_1, n_agents, n_chains, n_samples, output="map")
    _,map_agent_cpt_gamma_2 = process_params(gamma_cpt_2, n_agents, n_chains, n_samples, output="map")
    _,map_agent_cpt_gamma_3 = process_params(gamma_cpt_3, n_agents, n_chains, n_samples, output="map")

    plt.figure()
    plt.title("True vs. estimated Gamma (CPT)")
    plt.subplot(1,3,1)
    plt.scatter(gamma_cpt_true[:,0,0],map_agent_cpt_gamma_1)
    plt.plot(x,x)
    plt.ylabel("Estimated")
    plt.xlabel("True")
    plt.ylim([0,3])
    plt.xlim([0,3])
    plt.subplot(1,3,2)
    plt.scatter(gamma_cpt_true[:,0,0],map_agent_cpt_gamma_2)
    plt.ylabel("Estimated")
    plt.xlabel("True")
    plt.ylim([0,3])
    plt.xlim([0,3])
    plt.subplot(1,3,3)
    plt.scatter(gamma_cpt_true[:,0,0],map_agent_cpt_gamma_3)
    plt.ylabel("Estimated")
    plt.xlabel("True")
    plt.ylim([0,3])
    plt.xlim([0,3])

    # #--------PW----------#
    print("\nProcessing w...")
    plt.figure()
    # plt.subplot(1,2,1)
    plt.scatter(map_agent_cpt_delta_1,map_agent_cpt_gamma_1)
    plt.ylim([0,3])
    plt.xlim([0,3])
    # plt.subplot(1,2,2)
    plt.scatter(map_agent_cpt_delta_2,map_agent_cpt_gamma_2)
    # plt.ylim([0,3])
    # plt.xlim([0,3]) 
    plt.scatter(map_agent_cpt_delta_3,map_agent_cpt_gamma_3)
    # plt.ylim([0,3])
    # plt.xlim([0,3]) 
    plt.legend()

  
    plt.suptitle("Weighting function envolvment for CPT-species", fontsize=18)
    

    for i in range(n_agents): #agents
        plt.figure()
        w_1 = cpt_weighting_function(x, map_agent_cpt_delta_1[i],map_agent_cpt_gamma_1[i])
        w_2 = cpt_weighting_function(x, map_agent_cpt_delta_2[i],map_agent_cpt_gamma_2[i])
        w_3 = cpt_weighting_function(x, map_agent_cpt_delta_3[i],map_agent_cpt_gamma_3[i])
        w_true = cpt_weighting_function(x, delta_cpt_true[i,0,0], gamma_cpt_true[i,0,0])

        plt.subplot(1,3,1)
        plt.plot(x,w_1, label="Estimated")
        plt.plot(x,w_true, label="True")
        plt.plot(x,x, label = "No weighting")
        plt.xlim([0,1])
        plt.legend()
        plt.subplot(1,3,2)
        plt.plot(x,w_2, label="Estimated")
        plt.plot(x,w_true, label="True")
        plt.plot(x,x, label = "No weighting")
        plt.legend()
        plt.subplot(1,3,3)
        plt.plot(x,w_3, label="Estimated")
        plt.plot(x,w_true, label="True")
        plt.plot(x,x, label = "No weighting")
        plt.xlim([0,1])
        plt.legend()

    print("\nPlotting...")
    plt.show()

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
    # plt.subplot(1,3,1)
    plt.scatter(map_agent_lml_delta_1,map_agent_lml_gamma_1, label="Chunk 1")
    plt.ylim([0.7,1])
    plt.xlim([0.85,1.1])
    # plt.subplot(1,3,1)
    plt.scatter(map_agent_lml_delta_2,map_agent_lml_gamma_2, label="Chunk 2")
    # plt.ylim([0.7,1.1])
    # plt.xlim([0.7,1.1])
    # plt.subplot(1,3,3)    
    plt.scatter(map_agent_lml_delta_3,map_agent_lml_gamma_3, label="Chunk 3")
    # plt.ylim([0.7,1.1])
    # plt.xlim([0.7,1.1])
    plt.xlabel("Delta")
    plt.ylabel("Gamma")
    plt.legend()
    
    for i in range(n_agents): #agents
        plt.figure()
        w_1 = cpt_weighting_function(x, map_agent_lml_delta_1[i],map_agent_lml_gamma_1[i])
        w_2 = cpt_weighting_function(x, map_agent_lml_delta_2[i],map_agent_lml_gamma_2[i])
        w_3 = cpt_weighting_function(x, map_agent_lml_delta_3[i],map_agent_lml_gamma_3[i])
        w_true_1 = lml_weighting_function(x,10)
        w_true_2 = lml_weighting_function(x,50)
        w_true_3 = lml_weighting_function(x,100)

        plt.subplot(1,3,1)
        plt.plot(x,w_1, label="Estimated")
        plt.plot(x,w_true_1, label="Predicted")
        plt.plot(x,x, label = "No weighting")
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.legend()
        plt.subplot(1,3,2)
        plt.plot(x,w_2, label="Estimated")
        plt.plot(x,w_true_2, label="Predicted")
        plt.plot(x,x, label = "No weighting")
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.legend()
        plt.subplot(1,3,3)
        plt.plot(x,w_3, label="Estimated")
        plt.plot(x,w_true_3, label="True")
        plt.plot(x,x, label = "No weighting")
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.legend()
    
    print("\nPlotting...")
    plt.show()









sys.exit()

show_cpt = True
show_lml = True

if show_cpt:
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
    
    plt.figure()
    for c in range(n_chunks):
        plt.subplot(1,5,c+1)
        plt.scatter(map_agent_gamma[c], gamma_true[:,0,0])
        plt.xlim([0,2])
        plt.ylim([0,2])
        corr,_ = sc.pearsonr(map_agent_gamma[c],gamma_true[:,0,0])
        print("Pearson correlation coefficient for Gamma: ", corr)


    #--------PW----------#
    print("\nProcessing w...")
    plt.figure()
    for c in range(n_chunks):
        plt.subplot(1,5,c+1)
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

if show_lml:
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
            w_true.append(lml_weighting_function(x,(c+1)*50))

            plt.plot(x,w[i], label="Estimated")
            plt.plot(x,w_true[i], label="Predicted")
            plt.ylabel("Probability weight - $w(x)$", fontsize=14)
            plt.xlabel("Probability - $p(x)$", fontsize=14)
            plt.legend()

    print("\nPlotting...")
    plt.show()