#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import seaborn as sns
from scipy.special import gamma
from scipy.optimize import minimize
#%%
'''
Question 1 - Part 1
Obtain the sampling distribution of the three estimators for all three models for
n = 20. Compute the mean and variance of the estimators based on the sampling 
distribution. Use at least 5000 replications. 
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

def normal(n, iter=5000):
    capital_Ss = []
    means = []
    trimmed_Ss = []
    for i in range(iter):
        x = np.random.normal(0,1,n)
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
ax[1].set_title('Sampling Distribution of Mean')
ax[2].hist(trimmed_Ss, bins=50, density=True)
ax[2].set_title('Sampling Distribution of Trimmed S')
plt.suptitle('Sampling Distribution of Estimators for N(0,1)')
plt.show()

def mixture_of_normals(var1, var2, n, iter=5000):
    capital_Ss = []
    means = []
    trimmed_Ss = []
    for i in range(iter):
        first = 0.9*(np.random.normal(0, var1, n))
        second = 0.1*(np.random.normal(0, var2, n))
        x = first + second
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
ax[1].set_title('Sampling Distribution of Mean')
ax[2].hist(trimmed_Ss, bins=50, density=True)
ax[2].set_title('Sampling Distribution of Trimmed S')
plt.suptitle('Sampling Distribution of Estimators for 0.9N(0,1) + 0.1N(0,3)')
plt.show()


def normal_plus_cauchy(var1, var2, iter=5000):
    capital_Ss = []
    means = []
    trimmed_Ss = []
    for i in range(iter):
        first = 0.95*(np.random.normal(0, var1, n))
        second = 0.05*(np.random.standard_cauchy(n))
        x = first + second
        capital_S, mean, trimmed_S = myEstimators(n,x)
        capital_Ss.append(capital_S)
        means.append(mean)
        trimmed_Ss.append(trimmed_S)
    return capital_Ss, means, trimmed_Ss

capital_Ss, means, trimmed_Ss = normal_plus_cauchy(1, 1, n)
fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].hist(capital_Ss, bins=50, density=True)
ax[0].set_title('Sampling Distribution of Capital S')
ax[1].hist(means, bins=50, density=True)
ax[1].set_title('Sampling Distribution of Mean')
ax[2].hist(trimmed_Ss, bins=50, density=True)
ax[2].set_title('Sampling Distribution of Trimmed S')
plt.suptitle('Sampling Distribution of Estimators for 0.95N(0,1) + 0.05Cauchy(0,1)')
plt.show()




#%%
import warnings
warnings.filterwarnings('ignore')

# %%
'''
# Question 2
'''
rainfall_data = pd.read_excel('Rainfall DataSet - Assignment 2.xlsx', header=None)
print(len(rainfall_data))
print(rainfall_data.head())

# plotting the histogram of the rainfall data
sns.histplot(rainfall_data, bins=50)
plt.title('Histogram of Rainfall Data')
plt.xlabel('Rainfall')
plt.show()

# fitting a gamma distribution to the rainfall data
shape, loc, scale = stats.gamma.fit(rainfall_data)
print(shape, loc, scale)

# plotting the fitted gamma distribution
x = np.linspace(0, 3, 1000)
y = stats.gamma.pdf(x, shape, loc, scale)
sns.histplot(rainfall_data, bins=50, stat='density')
plt.plot(x, y, label='Fitted Gamma Distribution')
plt.title('Histogram of Rainfall Data')
plt.xlabel('Rainfall')
plt.legend()
plt.show()

# estimating the parameters using the method of moments
mean = np.mean(rainfall_data)
var = np.var(rainfall_data)
shape = mean**2/var
scale = var/mean
print("Shape:",shape[0])
print("Scale:",scale[0])

# estimating the parameters using the method of maximum likelihood using scipy
alpha, loc, beta = stats.gamma.fit(rainfall_data, floc=0)
print("Alpha:",alpha)
print("Beta:",beta)

#%%
shape_value, scale_value = shape[0], scale[0]

theretical_quantiles = stats.gamma.ppf(np.linspace(0.01, 0.99, 99), shape_value, loc=0, scale=scale_value)

observed_quantiles = np.sort(rainfall_data)

# plotting the qq plot
plt.scatter(theretical_quantiles, observed_quantiles)
plt.title('QQ Plot')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Observed Quantiles')
plt.show()





# %%
'''
# Assignment 2 - Part 2
'''
# drawing a sample of size 226 from the fitted gamma distribution
sample = stats.gamma.rvs(alpha, loc, beta, size=226)

# plotting the histogram of the sample
sns.histplot(sample, bins=50)
plt.title('Histogram of Sample')
plt.xlabel('Rainfall')
plt.show()

# finding the mle of the sample using scipy
alphas = []
betas = []
for i in range(1000):
    sample = stats.gamma.rvs(alpha, loc, beta, size=226)
    alpha_sample, loc_sample, beta_sample = stats.gamma.fit(sample, floc=0)
    alphas.append(alpha_sample)
    betas.append(beta_sample)


# plotting the histogram of the mle of the sample
sns.histplot(alphas, bins=50, color='red')
plt.title('Histogram of Alpha')
plt.xlabel('Alpha')
plt.show()

sns.histplot(betas, bins=50, color='green')
plt.title('Histogram of Beta')
plt.xlabel('Beta')
plt.show()


# %%
'''
# Assignment 2 - Part 3
'''
sample2 = stats.gamma.rvs(shape[0], loc, scale[0], size=226)
sns.histplot(sample2, bins=50)
plt.title('Histogram of Sample')
plt.xlabel('Rainfall')
plt.show()

# finding the mom estimators of the sample
shapes = []
scales = []
for i in range(1000):
    sample2 = stats.gamma.rvs(shape[0], loc, scale[0], size=226)
    mean_sample = np.mean(sample2)
    var_sample = np.var(sample2)
    shape_sample = mean_sample**2/var_sample
    scale_sample = var_sample/mean_sample
    shapes.append(shape_sample)
    scales.append(scale_sample)

# plotting the histogram of the mom estimators of the sample
sns.histplot(shapes, bins=50, color='red')
plt.title('Histogram of Shape')
plt.xlabel('Shape')
plt.show()

sns.histplot(scales, bins=50, color='green')
plt.title('Histogram of Scale')
plt.xlabel('Scale')
plt.show()

# %%
# comparing the mle and mom estimators using the bias and mse
print("Bias of Alpha MLE:", np.mean(alphas) - shape[0])
print("Bias of Beta MLE:", np.mean(betas) - scale[0])
print("Bias of Alpha MOM:", np.mean(shapes) - shape[0])
print("Bias of Beta MOM:", np.mean(scales) - scale[0])

print("MSE of Alpha MLE:", np.mean((np.array(alphas) - shape[0])**2))
print("MSE of Beta MLE:", np.mean((np.array(betas) - scale[0])**2))
print("MSE of Alpha MOM:", np.mean((np.array(shapes) - shape[0])**2))
print("MSE of Beta MOM:", np.mean((np.array(scales) - scale[0])**2))

# %%
