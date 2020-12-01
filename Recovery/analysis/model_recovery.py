from functions import read_output
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

z = read_output('model_comparison_LML_Gamble_2.mat','model_comparisson')

n_subjects = np.shape(z)[0]
n_samples = np.shape(z)[1]
n_chains = np.shape(z)[2]

print('---------------------------------------') 
print(f"Number of subjects = {n_subjects}")
print(f"Number of chains = {n_chains}")
print(f"Number of samples = {n_samples}")
print('---------------------------------------')

print(np.shape(z))

lml = 0
cpt = 0
z_choices2 = []
for i in range(n_subjects):
    for j in range(n_chains):
        for s in range(n_samples):
            z_choices2.append(z[i,s,j])
            if z[i,s,j] in [1,3,5,7]:
                # print(z[i,s,j])
                cpt += 1
            else:
                lml += 1
z_choices = [cpt/(n_subjects*n_samples*n_chains),lml/(n_subjects*n_samples*n_chains)]
print(z_choices)

x = ['CPT','LML']
plt.figure()
plt.suptitle("Modelindicator")
plt.bar(x,z_choices)
plt.ylim([0,1])
plt.show()