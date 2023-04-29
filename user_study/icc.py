import numpy as np
from scipy.stats import f
import pandas as pd

def calculate_icc(data):
    # Calculate the overall mean
    X_bar = np.mean(data)

    # Calculate the mean score for each observer
    obs_mean = np.mean(data, axis=1)

    # Calculate the between-observer variance
    SSB = np.sum(len(data[0]) * (obs_mean - X_bar)**2)

    # Calculate the within-observer variance
    SSW = 0
    for obs in data:
        SSW += np.sum((obs - np.mean(obs))**2)

    # Calculate the total variance
    SST = np.sum((data - X_bar)**2)

    # Calculate MSB and MSW
    k = len(obs_mean)
    N = len(data.flatten())
    MSB = SSB / (k - 1)
    MSW = SSW / (N - k)

    # Calculate ICC, F value, and p value
    ICC = (MSB - MSW) / (MSB + (k - 1) * MSW)
    F = ICC * ((N - k) / (k - 1))
    p_value = 1 - f.cdf(F, k - 1, N - k)

    return ICC, F, p_value

# Read data from a CSV file
data = pd.read_csv('user_icc3.csv', header=None).values
print(data)

# Calculate ICC
icc, f_value, p_value = calculate_icc(data)

# Print results
print('ICC:', icc)
print('F value:', f_value)
print('p value:', p_value)