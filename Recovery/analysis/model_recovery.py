from functions import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import matplotlib.pylab as pylab

params = {'axes.labelsize': 14,
         'axes.titlesize':16,
         'figure.max_open_warning': 0}
pylab.rcParams.update(params)

fig_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ),'..','..', 'Figures','tmp_figs'))

z_lml  = read_output('model_comparison_LML.mat'          ,'model_comparisson')
z_cpt  = read_output('model_comparison_CPT.mat'          ,'model_comparisson')
z_cpt2 = read_output('model_comparison_CPT_regular_S.mat','model_comparisson')

n_subjects = np.shape(z_cpt)[0]
n_samples = np.shape(z_cpt)[1]
n_chains = np.shape(z_cpt)[2]

print('---------------------------------------') 
print(f"Number of subjects = {n_subjects}")
print(f"Number of chains = {n_chains}")
print(f"Number of samples = {n_samples}")
print('---------------------------------------')

try:
    CPT_data = np.loadtxt(os.path.abspath(os.path.join(os.path.dirname( __file__ ),'..','samples_stats','CPT_int.txt')))

    intensities_cpt = []
    for i in range(20):
        tmp = [CPT_data[i*2],CPT_data[i*2+1]]
        intensities_cpt.append(tmp)

    LML_data = np.loadtxt(os.path.abspath(os.path.join(os.path.dirname( __file__ ),'..','samples_stats','LML_int.txt')))

    intensities_lml = []
    for i in range(10):
        tmp = [LML_data[i*2],LML_data[i*2+1]]
        intensities_lml.append(tmp)

    intensities = intensities_cpt + intensities_lml
    y_labels = ('1','10','11','20','21','30')
    labels = ['CPT','LML']
    cmap = plt.get_cmap('binary')
    fig, ax = plt.subplots()
    plot = ax.pcolormesh(range(len(labels)+1), range(31), intensities, cmap=cmap)
    ax.set_yticks([0.5,9.5,10.5,19.5,20.5,29.5])
    ax.set_yticklabels(y_labels)
    ax.set_xticks(np.arange(len(labels))+0.5)
    ax.set_xticklabels(labels)
    ax.set_xlim([0,2])
    ax.set_xlabel("'Ground Truth' model")
    ax.set_ylabel("Agent number")
    fig.colorbar(plot, ax=ax)
    plot.set_clim(0.0, 1.0)
    # plt.show()
    plt.savefig(os.path.join(fig_path,"model_recovery.png"))
except:
    print("______________________________ \nYou must run the choice-posteriors through the VBA_tool box!")
    z_choices_lml , z_choices_subject_lml  = model_select(z_lml , n_subjects, n_chains, n_samples)
    z_choices_cpt , z_choices_subject_cpt  = model_select(z_cpt , n_subjects, n_chains, n_samples)
    z_choices_cpt2, z_choices_subject_cpt2 = model_select(z_cpt2, n_subjects, n_chains, n_samples)
    print("\nLML_choices:\n")
    print(z_choices_subject_lml)
    print("\nCPT(inverse-s)_choices:\n")
    print(z_choices_subject_cpt)
    print("\nCPT(regular-s)_choices:\n")
    print(z_choices_subject_cpt2)

