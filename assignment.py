#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import seaborn as sns
from scipy.special import gamma
from scipy.optimize import minimize

#%%
import warnings
warnings.filterwarnings('ignore')

# %%
'''
# Question 2
'''
rainfall_data = pd.read_excel('Rainfall DataSet - Assignment 2.xlsx', header=None)

rainfall_data_flattened = np.array(rainfall_data).flatten()

# plotting the histogram of the rainfall data
plt.hist(rainfall_data_flattened, bins=30)
plt.title('Histogram of Rainfall Data')
plt.xlabel('Rainfall')
plt.show()

# fitting a gamma distribution to the rainfall data
shape, loc, scale = stats.gamma.fit(rainfall_data_flattened, floc=0)

# plotting the fitted gamma distribution
x = np.linspace(0, np.max(rainfall_data_flattened), 180)
y = stats.gamma.pdf(x, shape, loc, scale)
plt.hist(rainfall_data_flattened, bins=30, density=True, alpha=0.5)
plt.plot(x, y, label='Fitted Gamma Distribution')
plt.title('Histogram of Rainfall Data')
plt.xlabel('Rainfall')
plt.legend()
plt.show()

# estimating the parameters using the method of moments
mean = np.mean(rainfall_data_flattened)
var = np.var(rainfall_data_flattened)
shape = mean**2/var
scale = var/mean
print("Shape:",shape)
print("Scale:",scale)

# estimating the parameters using the method of maximum likelihood using scipy
alpha, loc, beta = stats.gamma.fit(rainfall_data_flattened, floc=0)
print("Alpha:",alpha)
print("Beta:",beta)

#%%
shape_value, scale_value = shape, scale
print(shape_value, scale_value)

# turning rainfall data to numpy array
rainfall_data = np.array(rainfall_data)

rainfall_data_flattened = rainfall_data.flatten()

# Create the Q-Q plot using stats.probplot
plt.figure(figsize=(8, 6))
res = stats.probplot(rainfall_data_flattened, dist=stats.gamma, sparams=(shape_value, 0, scale_value), plot=plt)
plt.title('Q-Q Plot with MOM')
plt.show()

#%%
shape_value, scale_value = alpha, beta
plt.figure(figsize=(8, 6))
res = stats.probplot(rainfall_data_flattened, dist=stats.gamma, sparams=(shape_value, 0, scale_value), plot=plt)
plt.title('Q-Q Plot with MLE')
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
sample2 = stats.gamma.rvs(shape, loc, scale, size=226)
sns.histplot(sample2, bins=50)
plt.title('Histogram of Sample')
plt.xlabel('Rainfall')
plt.show()

# finding the mom estimators of the sample
shapes = []
scales = []
for i in range(1000):
    sample2 = stats.gamma.rvs(shape, loc, scale, size=226)
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
print("Bias of Alpha MLE:", np.mean(alphas) - alpha)
print("Bias of Beta MLE:", np.mean(betas) - beta)
print("Bias of Alpha MOM:", np.mean(shapes) - shape)
print("Bias of Beta MOM:", np.mean(scales) - scale)


# using var plus bias squared to calculate mse
print("MSE of Alpha MLE:", np.var(alphas) + (np.mean(alphas) - alpha)**2)
print("MSE of Beta MLE:", np.var(betas) + (np.mean(betas) - beta)**2)
print("MSE of Alpha MOM:", np.var(shapes) + (np.mean(shapes) - shape)**2)
print("MSE of Beta MOM:", np.var(scales) + (np.mean(scales) - scale)**2)

# %%
