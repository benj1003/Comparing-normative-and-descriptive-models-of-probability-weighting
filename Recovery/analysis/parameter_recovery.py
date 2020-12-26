from functions import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import scipy.stats as sc
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.pylab as pylab

params = {'axes.labelsize': 14,
         'axes.titlesize':16,
         'figure.max_open_warning': 0}
pylab.rcParams.update(params)

fig_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ),'..','..', 'Figures','tmp_figs'))

print("------------------------------------")
print("          Parameter recovery")
print("------------------------------------")

cpt_parameter_recovery_files  = ['parameter_recovery_CPT_chunk_1.mat', 'parameter_recovery_CPT_chunk_2.mat','parameter_recovery_CPT_chunk_3.mat']
cpt_ground_truth_file         = "Choices_simulated_from_CPT.mat"
cpt2_parameter_recovery_files = ['parameter_recovery_CPT_regular_S_chunk_1.mat', 'parameter_recovery_CPT_regular_S_chunk_2.mat','parameter_recovery_CPT_regular_S_chunk_3.mat']
cpt2_ground_truth_file        = "Choices_simulated_from_CPT_regular_S.mat"

LML_parameter_recovery_files = ['parameter_recovery_LML_chunk_1.mat','parameter_recovery_LML_chunk_2.mat','parameter_recovery_LML_chunk_3.mat']

x = np.linspace(0,1,300)
x2 = np.linspace(0,10,1000)
n_chunks = len(cpt_parameter_recovery_files)

show_cpt   = True
show_lml   = False
show_plots = False

A      = [0,1,2,3]
marker = ['^','s','o','v']
colors = ['b','r','g']

if show_cpt:
    print("\n\n------------------------------------")
    print("Evaluating CPT data...")
    print("---\n")
    print("Reading output...")

    beta_cpt  = [None]*n_chunks
    delta_cpt = [None]*n_chunks
    gamma_cpt = [None]*n_chunks
    for c in range(n_chunks):
        _,beta_cpt[c],   delta_cpt[c],   gamma_cpt[c]    = read_output(cpt_parameter_recovery_files[c],'parameter_recovery')
    _,beta_cpt_true,delta_cpt_true,gamma_cpt_true = read_output(cpt_ground_truth_file,'parameter_recovery')

    beta_cpt2  = [None]*n_chunks
    delta_cpt2 = [None]*n_chunks
    gamma_cpt2 = [None]*n_chunks
    for c in range(n_chunks):
        _,beta_cpt2[c],   delta_cpt2[c],   gamma_cpt2[c]    = read_output(cpt2_parameter_recovery_files[c],'parameter_recovery')
    _,beta2_cpt_true,delta_cpt2_true,gamma_cpt2_true = read_output(cpt2_ground_truth_file,'parameter_recovery')

    n_agents = np.shape(beta_cpt[0])[0]
    n_samples = np.shape(beta_cpt[0])[1]
    n_chains = np.shape(beta_cpt[0])[2]

    print('---')
    print(f"Number of agents = {n_agents*2}")
    print(f"Number of chains = {n_chains}")
    print(f"Number of samples = {n_samples}")
    print(f"Number of chunks = {n_chunks}")
    print('---')

    #--------delta----------#
    print("\nProcessing Delta...")
    map_agent_cpt_delta = [None]*n_chunks
    map_agent_cpt2_delta = [None]*n_chunks
    for c in range(n_chunks):
        _,map_agent_cpt_delta[c]  = process_params(delta_cpt[c] , n_agents, n_chains, n_samples, output="map")
        _,map_agent_cpt2_delta[c] = process_params(delta_cpt2[c], n_agents, n_chains, n_samples, output="map")
        corr,_ = sc.pearsonr(np.concatenate((map_agent_cpt_delta[c],map_agent_cpt2_delta[c]),axis=0),np.concatenate((delta_cpt_true[:,0,0],delta_cpt2_true[:,0,0]),axis=0))
        print(f"Pearson correlation coefficient for Delta in chunk {c+1}: {corr:.3f}")

    plt.figure()
    plt.plot(x2,x2, color='lightgray', zorder=-1)
    for c in range(n_chunks):
        for i in range(n_agents):
            plt.scatter(delta_cpt_true[i,0,0] ,map_agent_cpt_delta[c][i] , marker='x', c=colors[c], s=80)
            plt.scatter(delta_cpt2_true[i,0,0],map_agent_cpt2_delta[c][i], marker='x', c=colors[c], s=80)
            plt.ylim([0,1.9])
            plt.xlim([0,1.9])
        #Dummy for legend
        plt.scatter(10,10,c='w',label=f"Chunk {c+1}")
    plt.xlabel("$\\delta_{'Ground-truth'}$")
    plt.ylabel("$\delta_e$")
    l = plt.legend(loc='upper left', handlelength=0)
    for text in l.get_texts():
        if text.get_text() == "Chunk 1": 
            text.set_color('b') 
        elif text.get_text() == "Chunk 2": 
            text.set_color('r') 
        elif text.get_text() == "Chunk 3": 
            text.set_color('g')
    plt.savefig(os.path.join(fig_path,"results-cpt-delta.png"))

    #--------gamma----------#
    print("\nProcessing Gamma...")
    map_agent_cpt_gamma  = [None]*n_chunks
    map_agent_cpt2_gamma = [None]*n_chunks
    for c in range(n_chunks):
        _,map_agent_cpt_gamma[c]  = process_params(gamma_cpt[c] , n_agents, n_chains, n_samples, output="map")
        _,map_agent_cpt2_gamma[c] = process_params(gamma_cpt2[c], n_agents, n_chains, n_samples, output="map")
        corr,_ = sc.pearsonr(np.concatenate((map_agent_cpt_gamma[c],map_agent_cpt2_gamma[c]),axis=0),np.concatenate((gamma_cpt_true[:,0,0],gamma_cpt2_true[:,0,0]),axis=0))
        print(f"Pearson correlation coefficient for Gamma in chunk {c+1}: {corr:.3f}")

    plt.figure()
    plt.plot(x2,x2, color='lightgray', zorder=-1)
    colors = ['b','r','g']
    for c in range(n_chunks):
        for i in range(n_agents):
            plt.scatter(gamma_cpt_true[i,0,0],map_agent_cpt_gamma[c][i] , marker='x', c=colors[c], s=80)
            plt.scatter(gamma_cpt2_true[i,0,0],map_agent_cpt2_gamma[c][i], marker='x', c=colors[c], s=80)
            plt.ylim([0,2.9])
            plt.xlim([0,2.9])
        #Dummies for legend
        plt.scatter(10,10,c='w',label=f"Chunk {c+1}")
    plt.xlabel("$\\gamma_{'Ground-truth'}$")
    plt.ylabel("$\gamma_e$")
    l = plt.legend(loc='upper left', handlelength=0)
    for text in l.get_texts():
        if text.get_text() == "Chunk 1": 
            text.set_color('b') 
        elif text.get_text() == "Chunk 2": 
            text.set_color('r') 
        elif text.get_text() == "Chunk 3": 
            text.set_color('g')
    plt.savefig(os.path.join(fig_path,"results-cpt-gamma.png"))

    sys.exit()
    # #--------PW----------#
    print("\nProcessing w...")
    plt.figure()
    for c in range(n_chunks):
        for i in range(n_agents):
            if i in A:
                plt.scatter(map_agent_cpt_gamma[c][i],map_agent_cpt_delta[c][i] , marker=marker[i], edgecolor=colors[c], facecolors='none', s=80)
                plt.scatter(map_agent_cpt2_gamma[c][i],map_agent_cpt2_delta[c][i], marker=marker[i], edgecolor=colors[c], facecolors='none', s=80)
            else:
                plt.scatter(map_agent_cpt_gamma[c][i],map_agent_cpt_delta[c][i] , marker='x', c=colors[c], s=80)
                plt.scatter(map_agent_cpt2_gamma[c][i],map_agent_cpt2_delta[c][i], marker='x', c=colors[c], s=80)
            plt.xlim([0,3.6])
            plt.ylim([0,1.9])
            plt.xticks([0,0.5,1,1.5,2,2.5])
        #Dummies for legend
        plt.scatter(10,10,c='w',label=f"Chunk {c+1}")
    plt.scatter(10,10, facecolors='none',label=" ")
    plt.scatter(10,10,label="Agent 1 & 11", marker=marker[0], edgecolor='k', facecolors='none')
    plt.scatter(10,10,label="Agent 2 & 12", marker=marker[1], edgecolor='k', facecolors='none')
    plt.scatter(10,10,label="Agent 3 & 13", marker=marker[2], edgecolor='k', facecolors='none')
    plt.scatter(10,10,label="Agent 4 & 14", marker=marker[3], edgecolor='k', facecolors='none')
    plt.scatter(10,10,marker='x',c='k',label="Other agents")
    plt.ylabel("$\\delta_e$")
    plt.xlabel("$\gamma_e$")
    l=plt.legend(loc='upper right')
    for text in l.get_texts():
        if text.get_text() == "Chunk 1": 
            text.set_color('b') 
        elif text.get_text() == "Chunk 2": 
            text.set_color('r') 
        elif text.get_text() == "Chunk 3": 
            text.set_color('g')
    plt.savefig(os.path.join(fig_path,"results-cpt-params.png"))
    
    #Zoom of above
    plt.figure(figsize=(5,4))
    for c in range(n_chunks):
        for i in range(n_agents):
            if i in A:
                plt.scatter(map_agent_cpt_gamma[c][i],map_agent_cpt_delta[c][i] , marker=marker[i], edgecolor=colors[c], facecolors='none', s=170)
                plt.scatter(map_agent_cpt2_gamma[c][i],map_agent_cpt2_delta[c][i], marker=marker[i], edgecolor=colors[c], facecolors='none', s=170)
            else:
                plt.scatter(map_agent_cpt_gamma[c][i],map_agent_cpt_delta[c][i] , marker='x', c=colors[c], s=170)
                plt.scatter(map_agent_cpt2_gamma[c][i],map_agent_cpt2_delta[c][i], marker='x', c=colors[c], s=170)
            plt.xlim([0.95,1.4])
            plt.ylim([1,1.2])
            plt.xticks([1,1.1,1.2,1.3,1.4])
            plt.yticks([1,1.05,1.1,1.15,1.2])
    plt.savefig(os.path.join(fig_path,"results-cpt-params-zoom.png"),bbox_inches='tight')

    w_1_diff = []
    w_2_diff = []
    w_3_diff = []

    for i in range(n_agents):
        plt.figure(figsize=(12,15))
        plt.suptitle(f"Probability Weighting function for CPT-Agent {i+1}", fontsize=18)
        w_true = cpt_weighting_function(x, delta_cpt_true[i,0,0], gamma_cpt_true[i,0,0])
        w = [None]*n_chunks
        for c in range(n_chunks):
            plt.subplot(3,2,(c*2)+1)
            if c == 0: plt.title("Parameter space", fontsize=14)
            plt.scatter(map_agent_cpt_gamma[c][i],map_agent_cpt_delta[c][i], edgecolor='b', facecolors='none', marker='^', label="Estimated", s=100)
            plt.scatter(gamma_cpt_true[i,0,0],delta_cpt_true[i,0,0], edgecolor='r',facecolor='none', label="'Ground truth'", s=100)
            
            plt.xlim([0,2.9])
            plt.ylim([0,1.9])
            plt.ylabel("$\\delta$")
            plt.xlabel("$\\gamma$")
            plt.legend(loc='upper right',prop={'size':8}, markerscale=0.7)
           
            w[c] = cpt_weighting_function(x, map_agent_cpt_delta[c][i],map_agent_cpt_gamma[c][i])

            if c == 0: 
                w_1_diff.append([a_i - b_i for a_i, b_i in zip(w_true,w[c])])
            elif c == 1:
                w_2_diff.append([a_i - b_i for a_i, b_i in zip(w_true,w[c])])
            else:
                w_3_diff.append([a_i - b_i for a_i, b_i in zip(w_true,w[c])])

            plt.subplot(3,2,(c*2)+2)
            if c == 0: plt.title("Weighting function", fontsize=14)
            plt.plot(x,w[c], 'b--', label="Estimated")
            plt.plot(x,w_true, 'r-.', label="'Ground truth'")
            plt.plot(x,x,'k-', label="No weighting")
            plt.xlabel("$\hat{p}(x)$",fontsize=14)
            plt.ylabel("$w(x)$",fontsize=14)
            plt.legend(loc='upper left', prop={'size':8})

        plt.subplots_adjust(wspace=0.3,hspace=0.3)
        plt.savefig(os.path.join(fig_path,f"results-cpt-w-agent{i+1}.png"))


    for i in range(n_agents):
        plt.figure(figsize=(12,15))
        plt.suptitle(f"Probability Weighting function for CPT-Agent {i+11}", fontsize=18)
        w_true = cpt_weighting_function(x, delta_cpt2_true[i,0,0], gamma_cpt2_true[i,0,0])
        w = [None]*n_chunks
        for c in range(n_chunks):
            plt.subplot(3,2,(c*2)+1)
            if c == 0: plt.title("Parameter space", fontsize=14)
            plt.scatter(map_agent_cpt2_gamma[c][i],map_agent_cpt2_delta[c][i], edgecolor='b', facecolors='none', marker='^', label="Estimated", s=100)
            plt.scatter(gamma_cpt2_true[i,0,0],delta_cpt2_true[i,0,0], edgecolor='r',facecolor='none', label="'Ground truth'", s=100)
            
            plt.xlim([0,2.9])
            plt.ylim([0,1.9])
            plt.ylabel("$\\delta$")
            plt.xlabel("$\\gamma$")
            plt.legend(loc='upper left', prop={'size':8}, markerscale=0.7)
           
            w[c] = cpt_weighting_function(x, map_agent_cpt2_delta[c][i],map_agent_cpt2_gamma[c][i])

            if c == 0: 
                w_1_diff.append([a_i - b_i for a_i, b_i in zip(w_true,w[c])])
            elif c == 1:
                w_2_diff.append([a_i - b_i for a_i, b_i in zip(w_true,w[c])])
            else:
                w_3_diff.append([a_i - b_i for a_i, b_i in zip(w_true,w[c])])

            plt.subplot(3,2,(c*2)+2)
            if c == 0: plt.title("Weighting function", fontsize=14)
            plt.plot(x,w[c], 'b--', label="Estimated")
            plt.plot(x,w_true, 'r-.', label="'Ground truth'")
            plt.plot(x,x,'k-', label="No weighting")
            plt.xlabel("$\hat{p}(x)$",fontsize=14)
            plt.ylabel("$w(x)$",fontsize=14)
            plt.legend(loc='upper left', prop={'size':8})

        plt.subplots_adjust(wspace=0.3,hspace=0.3)
        plt.savefig(os.path.join(fig_path,f"results-cpt-w-agent{i+11}.png"))

    diff = [w_1_diff,w_2_diff,w_3_diff]
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5,8))
    plt.setp(ax, xticks=[0, 150, 300], xticklabels=['0', '0.5', '1'], yticks=[-0.02, 0, 0.02])
    for c in range(n_chunks):
        df = pd.DataFrame(diff[c])
        df = pd.melt(frame = df, var_name = '$\hat{p}(x)$', value_name = '$\\bar{w}(x)-w(x)$')
        sns.lineplot(ax = ax[c],data = df,ci=95,x = '$\hat{p}(x)$', y = '$\\bar{w}(x)-w(x)$')
        ax[c].collections[0].set_label('95 pct. confidence interval')
        ax[c].hlines(y=0, xmin=0, xmax=len(x), color='k', linestyle='-')
        ax[c].set_title(f"Chunk {c+1}", fontsize = 14)
        ax[c].legend(loc='upper left')
        ax[c].set_ylim([-0.025,0.025])
        plt.tight_layout()
    plt.savefig(os.path.join(fig_path,"results-cpt-difference.png"))

    if show_plots:
        print("\nPlotting...")
        plt.show()

#-------------------------------------------------------------------------------------------------------------
#--------------------------------LML--------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

if show_lml:
    print("\n\n------------------------------------")
    print("Evaluating LML data...")
    print("---\n")
    print("Reading output...")

    beta_lml = [None]*n_chunks
    delta_lml = [None]*n_chunks
    gamma_lml = [None]*n_chunks
    for c in range(n_chunks):
        _,beta_lml[c],   delta_lml[c],   gamma_lml[c]    = read_output(LML_parameter_recovery_files[c],'parameter_recovery')

    n_agents = np.shape(beta_lml[0])[0]
    n_samples = np.shape(beta_lml[0])[1]
    n_chains = np.shape(beta_lml[0])[2]

    print('---')
    print(f"Number of agents = {n_agents}")
    print(f"Number of chains = {n_chains}")
    print(f"Number of samples = {n_samples}")
    print('---')

    #--------delta----------#
    print("\nProcessing Delta...")
    map_agent_lml_delta = [None]*n_chunks
    for c in range(n_chunks):
        _,map_agent_lml_delta[c] = process_params(delta_lml[c], n_agents, n_chains, n_samples, output="map")

    #--------gamma----------#
    print("\nProcessing Gamma...")
    map_agent_lml_gamma = [None]*n_chunks
    for c in range(n_chunks):
        _,map_agent_lml_gamma[c] = process_params(gamma_lml[c], n_agents, n_chains, n_samples, output="map")

    #--------PW----------#
    print("\nProcessing w...")
    plt.figure()
    for c in range(n_chunks):
        for i in range(n_agents):
            if i in A:
                plt.scatter(map_agent_lml_gamma[c][i],map_agent_lml_delta[c][i], marker=marker[i], edgecolor=colors[c], facecolors='none', s=50)
            else:
                plt.scatter(map_agent_lml_gamma[c][i],map_agent_lml_delta[c][i], marker='x', c=colors[c], s=50)
        plt.scatter(10,10,c='w',label=f"Chunk {c+1}")
    plt.ylabel("$\\delta_e$")
    plt.xlabel("$\gamma_e$")
    plt.xlim([0,2.9])
    plt.ylim([0,1.9])
    plt.scatter(10,10, facecolors='none',label=" ")
    plt.scatter(10,10, marker=marker[0], edgecolor='k', facecolors='none', label=f"Agent 1")
    plt.scatter(10,10, marker=marker[1], edgecolor='k', facecolors='none', label=f"Agent 2")
    plt.scatter(10,10, marker=marker[2], edgecolor='k', facecolors='none', label=f"Agent 3")
    plt.scatter(10,10, marker=marker[3], edgecolor='k', facecolors='none', label=f"Agent 4")
    plt.scatter(10,10,marker='x',c='k',label="Agents 5-10")
    l=plt.legend(loc='upper right')
    for text in l.get_texts():
        if text.get_text() == "Chunk 1": 
            text.set_color('b') 
        elif text.get_text() == "Chunk 2": 
            text.set_color('r') 
        elif text.get_text() == "Chunk 3": 
            text.set_color('g')
    plt.savefig(os.path.join(fig_path,"results-lml-params.png"))

    #Zoom of above
    plt.figure(figsize=(5,4))
    for c in range(n_chunks):
        for i in range(n_agents):
            if i in A:
                plt.scatter(map_agent_lml_gamma[c][i],map_agent_lml_delta[c][i], marker=marker[i], edgecolor=colors[c], facecolors='none', s=110)
            else:
                plt.scatter(map_agent_lml_gamma[c][i],map_agent_lml_delta[c][i], marker='x', c=colors[c], s=110)
        plt.scatter(10,10,c='w',label=f"Chunk {c+1}")
    plt.xlim([0.75,1])
    plt.ylim([0.94,1.05])
    plt.xticks([0.75,0.8,0.85,0.9,0.95,1])
    plt.yticks([0.94,0.96,0.98,1,1.02,1.04])
    plt.savefig(os.path.join(fig_path,"results-lml-params-zoom.png"), bbox_inches='tight')

    w_1_diff = []
    w_2_diff = []
    w_3_diff = []

    chunk_len = [10,50,100]
    w_true = [None]*n_chunks
    for c in range(n_chunks):
        w_true[c] = lml_weighting_function(x,chunk_len[c])
    colors = ['b','r','g']
    for i in range(n_agents): 
        plt.figure(figsize=(12,15))
        plt.suptitle(f"Probability Weighting function for LML-Agent {i+1}", fontsize=18)
        w = [None]*n_chunks
        for c in range(n_chunks):
            plt.subplot(3,2,(c*2)+1)
            if c == 0: plt.title(f"Parameter space",fontsize=14)
            plt.scatter(map_agent_lml_gamma[c][i],map_agent_lml_delta[c][i], edgecolor='b', facecolors='none', marker='^', label="Estimated", s=100)
            plt.xlim([0,2.4])
            plt.ylim([0,1.6])
            plt.ylabel("$\\delta$")
            plt.xlabel("$\\gamma$")
            plt.legend(loc='upper left', prop={'size':8}, markerscale=0.7)
            w[c] = cpt_weighting_function(x, map_agent_lml_delta[c][i],map_agent_lml_gamma[c][i])
            if c == 0: 
                w_1_diff.append([a_i - b_i for a_i, b_i in zip(w_true[c],w[c])])
            elif c == 1:
                w_2_diff.append([a_i - b_i for a_i, b_i in zip(w_true[c],w[c])])
            else:
                w_3_diff.append([a_i - b_i for a_i, b_i in zip(w_true[c],w[c])])
            plt.subplot(3,2,(c*2)+2)
            if c == 0: plt.title("Weighting function", fontsize=14)
            plt.plot(x,w[c], 'b--', label="Estimated")
            plt.plot(x,w_true[c],'r-.', label="Predicted")
            plt.plot(x,x,'k-', label = "No weighting")
            plt.xlabel("$\hat{p}(x)$")
            plt.ylabel("$w(x)$")
            plt.legend(loc='upper left', prop={'size':8})
        plt.subplots_adjust(wspace=0.3,hspace=0.3)
        plt.savefig(os.path.join(fig_path,f"results-lml-w-agent{i+1}.png"))

    diff = [w_1_diff,w_2_diff,w_3_diff]
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5,8))
    plt.setp(ax, xticks=[0, 150, 300], xticklabels=['0', '0.5', '1'], yticks=[-0.02, 0,0.02])
    for c in range(n_chunks):
        df = pd.DataFrame(diff[c])
        df = pd.melt(frame = df, var_name = '$\hat{p}(x)$', value_name = '$\\bar{w}(x)-w(x)$')
        sns.lineplot(ax = ax[c],data = df,ci=95,x = '$\hat{p}(x)$', y = '$\\bar{w}(x)-w(x)$')
        ax[c].collections[0].set_label('95 pct. confidence interval')
        ax[c].hlines(y=0, xmin=0, xmax=len(x), color='k', linestyle='-')
        ax[c].set_title(f"Chunk {c+1}", fontsize=14)
        ax[c].set_ylim([-0.025,0.025])
        ax[c].legend(loc='upper left')
        ax[c].set_ylabel(" ")
        plt.tight_layout()
    plt.savefig(os.path.join(fig_path,"results-LML-difference.png"))

    if show_plots:
        print("\nPlotting...")
        plt.show()

print("------------------------------------")