#%%
'''
Question 1 - Part 1
Obtain the sampling distribution of the three estimators for all three models for
n = 20. Compute the mean and variance of the estimators based on the sampling 
distribution. Use at least 5000 replications. 
'''

import numpy as np
import matplotlib.pyplot as plt


def myEstimators(n,x):

    # sample standard deviation S
    divisor = 1/(n - 1)
    capital_S = np.sqrt(divisor * np.sum((x - np.mean(x))**2))

    # mean deviation
    mean = np.mean(abs(x - np.mean(x)))

    # trimmed standard deviation
    # discarding the smallest 0.1n/2 and largest 0.1n/2
    new_x = np.sort(x)
    new_x = new_x[int(0.1*n/2):int(n - 0.1*n/2)]
    trimmed_S = np.sqrt(divisor * np.sum((new_x - np.mean(new_x))**2))

    return capital_S, mean, trimmed_S



# obtaining the sampling distribution of the estimators
n = 20

def normal(n, iter=5000):

    # creating lists to store the estimators
    capital_Ss = []
    means = []
    trimmed_Ss = []

    for i in range(iter):

        # drawing a sample of size n from N(0,1)
        x = np.random.normal(0,1,n)

        # computing the estimators
        capital_S, mean, trimmed_S = myEstimators(n,x)
        
        capital_Ss.append(capital_S)
        means.append(mean)
        trimmed_Ss.append(trimmed_S)
    return capital_Ss, means, trimmed_Ss

capital_Ss, means, trimmed_Ss = normal(n)

# plotting the sampling distribution of the estimators
fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].hist(capital_Ss, bins=50, density=True)
ax[0].set_title('Sampling Distribution of Capital S')
ax[1].hist(means, bins=50, density=True)
ax[1].set_title('Sampling Distribution of Mean Deviations')
ax[2].hist(trimmed_Ss, bins=50, density=True)
ax[2].set_title('Sampling Distribution of Trimmed S')
plt.suptitle('Sampling Distribution of Estimators for N(0,1)')
plt.show()

def mixture_of_normals(var1, var2, n, iter=5000):
    capital_Ss = []
    means = []
    trimmed_Ss = []
    for i in range(iter):

        # selecting N(0,1) with probability 0.9 and N(0,3) with probability 0.1
        epsilon = 0.1
        if np.random.uniform() < epsilon:
            x = np.random.normal(0, var2, n)
        else:
            x = np.random.normal(0, var1, n)

        capital_S, mean, trimmed_S = myEstimators(n,x)
        capital_Ss.append(capital_S)
        means.append(mean)
        trimmed_Ss.append(trimmed_S)
    return capital_Ss, means, trimmed_Ss

capital_Ss, means, trimmed_Ss = mixture_of_normals(1, 3, n)
fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].hist(capital_Ss, bins=50, density=True)
ax[0].set_title('Sampling Distribution of Capital S')
ax[1].hist(means, bins=50, density=True)
ax[1].set_title('Sampling Distribution of Mean Deviations')
ax[2].hist(trimmed_Ss, bins=50, density=True)
ax[2].set_title('Sampling Distribution of Trimmed S')
plt.suptitle('Sampling Distribution of Estimators for 0.9N(0,1) + 0.1N(0,3)')
plt.show()


def normal_plus_cauchy(var1, iter=5000):
    capital_Ss = []
    means = []
    trimmed_Ss = []
    for i in range(iter):

        epsilon = 0.05
        if np.random.uniform() < epsilon:
            x = np.random.standard_cauchy(n)
        else:
            x = np.random.normal(0, var1, n)
        
        capital_S, mean, trimmed_S = myEstimators(n,x)
        capital_Ss.append(capital_S)
        means.append(mean)
        trimmed_Ss.append(trimmed_S)

    return capital_Ss, means, trimmed_Ss

capital_Ss, means, trimmed_Ss = normal_plus_cauchy(1, n)
fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].hist(capital_Ss, bins=50, density=True)
ax[0].set_title('Sampling Distribution of Capital S')
ax[1].hist(means, bins=50, density=True)
ax[1].set_title('Sampling Distribution of Mean Deviations')
ax[2].hist(trimmed_Ss, bins=50, density=True)
ax[2].set_title('Sampling Distribution of Trimmed S')
plt.suptitle('Sampling Distribution of Estimators for 0.95N(0,1) + 0.05Cauchy(0,1)')
plt.show()
# %%

# computing the bias and mse of the estimators
def bias(estimator, true_value):
    return np.mean(estimator) - true_value

def mse(estimator, true_value):
    return np.var(estimator) + (np.mean(estimator) - true_value)**2

# for normal
print("Normal Distribution")
capital_Ss, means, trimmed_Ss = normal(n)
print("Bias of Capital S:", bias(capital_Ss, 1))
print("Bias of Mean Deviations:", bias(means, 1))
print("Bias of Trimmed S:", bias(trimmed_Ss, 1))
print("MSE of Capital S:", mse(capital_Ss, 1))
print("MSE of Mean Deviations:", mse(means, 1))
print("MSE of Trimmed S:", mse(trimmed_Ss, 1))
print()

# for mixture of normals
print("Mixture of Two Normals")
capital_Ss, means, trimmed_Ss = mixture_of_normals(1, 3, n)
print("Bias of Capital S:", bias(capital_Ss, 1))
print("Bias of Mean Deviations:", bias(means, 1))
print("Bias of Trimmed S:", bias(trimmed_Ss, 1))
print("MSE of Capital S:", mse(capital_Ss, 1))
print("MSE of Mean Deviations:", mse(means, 1))
print("MSE of Trimmed S:", mse(trimmed_Ss, 1))
print()

# for normal plus cauchy
print("Normal Plus Cauchy")
capital_Ss, means, trimmed_Ss = normal_plus_cauchy(1, n)
print("Bias of Capital S:", bias(capital_Ss, 1))
print("Bias of Mean Deviations:", bias(means, 1))
print("Bias of Trimmed S:", bias(trimmed_Ss, 1))
print("MSE of Capital S:", mse(capital_Ss, 1))
print("MSE of Mean Deviations:", mse(means, 1))
print("MSE of Trimmed S:", mse(trimmed_Ss, 1))
print()

# %%
