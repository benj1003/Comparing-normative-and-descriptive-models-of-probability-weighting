import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

N = 100000
N_v = 500
step = 0.001

p = np.arange(0,1+step,step)
v = list(np.linspace(0,1.5,N_v))

lambda_locs = [0.4,0.6,1]
delta_locs = [0.8,0.9,1]

intensities = []

for k in range(len(lambda_locs)):
    _lambda = []
    _delta = []
    w_out = []
    for i in range(N):
        _lambda.append(np.random.normal(loc=lambda_locs[k],scale=0.1))
        _delta.append(np.random.normal(loc=delta_locs[k],scale=0.1))

    #Parameter estimates
    bins_l, edges_l = np.histogram(_lambda, bins=v)
    bins_d, edges_d = np.histogram(_delta, bins=v)

    t1 = ((np.array([bins_l]))).T
    t2 = (np.array([bins_d]))

    X = edges_l[1:]
    Y = edges_d[1:]
    Z = t1 @ t2

    plt.figure()
    plt.pcolormesh(X,Y,Z)
    plt.xlabel("$\lambda$")
    plt.ylabel("$\delta$")
    

    # #Weighting functions
    bins_w = []
    edges_w = []
    for i in range(len(p)):
        w = [] 
        for j in range(N):
            tmp = _delta[j]*(p[i]**_lambda[j]) + (1-p[i])**_lambda[j]
            w.append(_delta[j]*p[i]**_lambda[j]/tmp)
        w_out.append(w)
        bins_tmp, edges_tmp = np.histogram(w_out[i], bins=v)

        bins_w.append(bins_tmp)
        edges_w.append(edges_tmp[1:])

    tmp = np.array(bins_w)

    for i in range(np.shape(tmp)[0]*np.shape(tmp)[1]):
        if tmp.flat[i] > 5000:
            tmp.flat[i] = 5000 

    intensities.append(tmp.T)

plt.figure()
plt.pcolormesh(p,v[1:],intensities[0])
plt.ylim([0,1.05])
plt.yticks([0,0.5,1])
plt.ylabel("$w(p)$")
plt.xlabel("$p$")
plt.figure()
plt.pcolormesh(p,v[1:],intensities[1])
plt.ylim([0,1.05])
plt.yticks([0,0.5,1])
plt.ylabel("$w(p)$")
plt.xlabel("$p$")
plt.figure()
plt.pcolormesh(p,v[1:],intensities[2])
plt.ylim([0,1.05])
plt.yticks([0,0.5,1])
plt.ylabel("$w(p)$")
plt.xlabel("$p$")
plt.show()