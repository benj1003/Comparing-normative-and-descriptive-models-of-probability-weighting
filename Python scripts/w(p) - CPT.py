import matplotlib.pyplot as plt
import numpy as np 

step = 0.001
p = np.arange(0,1+step,step)
tmp_alpha = -2
tmp_gamma = -2

gammas = [0.2,0.6,0.6]
alphas = [0.8,0.8,0.8]

# plt.figure()
for i in range(len(gammas)):
    gamma = gammas[0]
    alpha = alphas[0]
    w = []
    for i in p:
        tmp = alpha*(i**gamma) + (1-i)**gamma
        w.append(alpha*i**gamma/tmp)
    
    gamma = gammas[2]
    alpha = alphas[2]
    w2 = []
    for i in p:
        tmp = alpha*(i**gamma) + (1-i)**gamma
        w2.append(alpha*i**gamma/tmp)

# plt.figure()
# plt.plot(p,p, 'k-', label="Linear/no weighting")
# plt.plot(p,w, 'r--', label="Inverse-S")
# plt.plot(p,w2, 'b-.', label="S-shape")
# plt.legend()
# # plt.title(f"\alpha = {alpha}, \gamma = {gamma}")
# plt.ylabel("Decistion weight - $w(x)$", fontsize=14)
# plt.xlabel("Probability - $p(x)$", fontsize=14)
# plt.xticks([0,0.5,1])
# plt.yticks([0,0.5,1])
# plt.show()

###For dynamic
plt.figure()
plt.plot(p,p, 'k-')
plt.plot(p,w, 'r--')
plt.plot(p,w2, 'b-.')
plt.ylabel("Decistion weight - $w(x)$", fontsize=14)
plt.xlabel("Probability - $p(x)$", fontsize=14)
plt.xticks([0,0.5,1])
plt.yticks([0,0.5,1])
plt.show()
