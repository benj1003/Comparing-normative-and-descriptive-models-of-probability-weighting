import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

alpha = 2
mu = 0
sigma = 1
x = np.linspace(mu - 3*sigma*alpha, mu + 3*sigma*alpha, 100000)

plt.plot(x, stats.norm.pdf(x, mu, sigma),'k-',label='$p(x)$')
plt.plot(x, stats.norm.pdf(x, mu, sigma*alpha),'k--', label='$w(x)$')
plt.xlim([-4.5,4.5])
plt.xticks([-4,-2,0,2,4])
plt.yticks([0.1,0.2,0.3,0.4])
plt.legend(loc='upper left')
plt.xlabel("x")
plt.ylabel("PDF")
plt.show()

plt.plot(x, stats.norm.cdf(x, mu, sigma),'k-',label='$F_p(x)$')
plt.plot(x, stats.norm.cdf(x, mu, sigma*alpha),'k--', label='$F_w(x)$')
plt.xlim([-4.5,4.5])
plt.xticks([-4,-2,0,2,4])
plt.yticks([0.2,0.4,0.6,0.8,1.0])
plt.legend(loc='upper left')
plt.xlabel("x")
plt.ylabel("CDF")
plt.show()

plt.plot(stats.norm.cdf(x, mu, sigma), stats.norm.cdf(x, mu, sigma),'k-',label='$F_p(F_p(x))$')
plt.plot(stats.norm.cdf(x, mu, sigma), stats.norm.cdf(x, mu, sigma*alpha),'k--', label='$F_w(F_p(x))$')
plt.xlim([-0.05,1.05])
plt.xticks([0,0.2,0.4,0.6,0.8,1.0])
plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
plt.legend(loc='upper left')
plt.xlabel("CDF $F_p(x)$")
plt.ylabel("CDF")
plt.show()