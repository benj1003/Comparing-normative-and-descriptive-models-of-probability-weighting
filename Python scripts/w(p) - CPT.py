import matplotlib.pyplot as plt
import numpy as np 

step = 0.001
p = np.arange(0,1+step,step)
tmp_alpha = -2
tmp_gamma = -2

gammas = [0.4,0.4,0.4]
alphas = [0.2,1,5]

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
# for j in range(20):
#     gamma = 0.2*(j+1)
#     w = []
#     for i in p:
#         tmp = ((i**gamma) + (1-i)**gamma)**(1/gamma)
#         w.append(i**gamma/tmp)

    # plt.figure()
    # plt.scatter(alpha,gamma,marker = 'x')
    # # plt.scatter(tmp_alpha, tmp_gamma, alpha = 0.5)
    # plt.xlim([0,1.2])
    # plt.ylim([0,1.2])
    # plt.xticks([0,0.5,1])
    # plt.yticks([0,0.5,1])
    # tmp_alpha = alpha
    # tmp_gamma = gamma
# plt.subplot(4,5,j+1)
plt.figure()
plt.plot(p,p, 'k-', label="Linear/no weighting")
plt.plot(p,w, 'k-', label="Inverse-S")
plt.plot(p,w2, 'k-.', label="S-shape")
plt.legend()
# plt.title(f"\alpha = {alpha}, \gamma = {gamma}")
plt.ylabel("Decistion weight - $w(x)$", fontsize=14)
plt.xlabel("Probability - $p(x)$", fontsize=14)
plt.xticks([0,0.5,1])
plt.yticks([0,0.5,1])
plt.show()
