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

