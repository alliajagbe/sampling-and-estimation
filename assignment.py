#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
#%%
'''
Question 1
'''

def myEstimators(n,x):
    divisor = 1/(n - 1)
    capital_S = np.sqrt(divisor * np.sum((x - np.mean(x))**2))
    mean = np.mean(abs(x - np.mean(x)))
    # discarding the smallest 0.1n/2 and largest 0.1n/2
    new_x = np.sort(x)
    new_x = new_x[int(0.1*n/2):int(n - 0.1*n/2)]
    trimmed_S = np.sqrt(divisor * np.sum((new_x - np.mean(new_x))**2))

    return capital_S, mean, trimmed_S



# obtaining the sampling distribution of the estimators
n = 20
capital_Ss = []
means = []
trimmed_Ss = []
for i in range(5000):
    x = np.random.normal(0,1,n)
    capital_S, mean, trimmed_S = myEstimators(n,x)
    capital_Ss.append(capital_S)
    means.append(mean)
    trimmed_Ss.append(trimmed_S)

# plotting the sampling distribution of the estimators
fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].hist(capital_Ss, bins=50, density=True)
ax[0].set_title('Sampling Distribution of Capital S')
ax[1].hist(means, bins=50, density=True)
ax[1].set_title('Sampling Distribution of Mean')
ax[2].hist(trimmed_Ss, bins=50, density=True)
ax[2].set_title('Sampling Distribution of Trimmed S')
plt.suptitle('Sampling Distribution of Estimators for N(0,1)')
plt.show()



# %%
