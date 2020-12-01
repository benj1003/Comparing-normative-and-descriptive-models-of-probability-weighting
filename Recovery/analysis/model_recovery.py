from functions import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

z_lml = read_output('model_comparison_LML_Gamble_2.mat','model_comparisson')
z_cpt = read_output('model_comparison_CPT_Gamble_2.mat','model_comparisson')

print(z_lml)
print()
print(z_cpt)

sys.exit()
n_subjects = np.shape(z_lml)[0]
n_samples = np.shape(z_lml)[1]
n_chains = np.shape(z_lml)[2]

print('---------------------------------------') 
print(f"Number of subjects = {n_subjects}")
print(f"Number of chains = {n_chains}")
print(f"Number of samples = {n_samples}")
print('---------------------------------------')

#Running on lml_data
z_choices_lml, z_choices_subject_lml = model_select(z_lml, n_subjects, n_chains, n_samples)
z_choices_cpt, z_choices_subject_cpt = model_select(z_cpt, n_subjects, n_chains, n_samples)


print(z_choices_lml)
#Running on lml_data
print(z_choices_cpt)

#Raw plot
labels = ['CPT-LML','LML-LML']
plt.figure()
plt.suptitle("Modelindicator")
plt.bar(labels,z_choices_lml)
plt.ylim([0,1])

labels = ['CPT-CPT','LML-CPT']
plt.figure()
plt.suptitle("Modelindicator")
plt.bar(labels,z_choices_cpt)
plt.ylim([0,1])

plt.show()

intensities = z_choices_subject_lml + z_choices_subject_cpt
#Heatmap
labels = ['CPT','LML']
cmap = plt.get_cmap('binary')
fig, ax = plt.subplots()
plot = ax.pcolormesh(range(len(labels)+1), range(100), intensities, cmap=cmap)
ax.set_yticks([1,25,50,51,75,100])
ax.set_yticklabels(['1','25','50','1','25','50'],fontsize = 10)
ax.set_xticks(np.arange(len(labels))+0.5)
ax.set_xticklabels(labels)
ax.set_xlim([0,2])
ax.set_title("Posterior model probabilities", fontsize = 14)
fig.colorbar(plot, ax=ax)
plot.set_clim(0.0, 1.0)
plt.show()