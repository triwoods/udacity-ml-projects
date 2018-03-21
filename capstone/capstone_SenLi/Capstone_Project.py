
# coding: utf-8

# Import modules
# 

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time


# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


# In[3]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from scipy.signal import freqz, convolve

import utils as u

plt.ion()


# Loading training and testing data

# In[4]:


X = u.q15_to_lsf(np.load('./data/train/line_spectrum_frequency_low_band_speech.npy'))
y = u.q15_to_lsf(np.load('./data/train/line_spectrum_frequency_high_band_speech.npy'))

X_test = u.q15_to_lsf(np.load('./data/test/line_spectrum_frequency_low_band_speech.npy'))
y_test = u.q15_to_lsf(np.load('./data/test/line_spectrum_frequency_high_band_speech.npy'))

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.05, random_state=0)

print("Input Dimension: %d" % (X_train.shape[1]))
print("Output Dimension: %d" % (y_train.shape[1]))

print("Total Training data %d" % (X_train.shape[0]))
print("Total Validating data %d" % (X_valid.shape[0]))
print("Total Testing data %d" % (X_test.shape[0]))


# In[5]:


u.plot_lsf(X_train[100:104,:])    


# In[6]:


u.plot_lsf_hist(X_train)


# In[7]:


def train_validate(learner, sample_size, X_train, y_train, X_valid, y_valid):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_valid: features validating set
       - y_valid: income validating set
    '''
    sample_size = int(sample_size)

    print(sample_size)
    
    results = {}

    start = time() # Get start time
    X_train = X_train[:sample_size, :]
    y_train = y_train[:sample_size, :]
    
    learner = learner.fit(X_train, y_train)
    end = time() # Get end time
    
    print('training_done!')

    results['train_time'] = end - start

    start = time() # Get start time
    predictions_valid = learner.predict(X_valid)
    
    print('validating_done!')
    
    print(predictions_valid.shape)
    predictions_train = learner.predict(X_train[:1000, :])
    print(predictions_train.shape)
    end = time() # Get end time

    # Calculate the total prediction time
    results['pred_time'] = end - start

    print(np.argwhere(np.isnan(predictions_train)))
    print(np.argwhere(np.isnan(predictions_valid)))
    
    print(predictions_train)
    print(predictions_valid)
    
    # Compute accuracy on the first 3000 training samples
    # results['error_train'] = mean_squared_error(y_train[:1000, :], predictions_train, multioutput='raw_values')

    # Compute accuracy on validation set
    # results['error_valid'] = mean_squared_error(y_valid, predictions_valid, multioutput='raw_values')

    # TODO: Compute F-score on the the first 300 training samples
    # results['r2_train'] = r2_score(y_train, predictions_train)

    # TODO: Compute F-score on the test set
    # results['r2_valid'] = r2_score(y_valid, predictions_valid)

    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    # Return the results
    return results


# In[8]:


clf_benchmark_1 = LinearRegression(n_jobs=1)
# clf_benchmark_2 = DecisionTreeRegressor()
# clf_benchmark_3 = RandomForestRegressor(random_state=0, n_jobs=-1)


# In[9]:


num_train_data = X_train.shape[0]
samples_1 = num_train_data // 10
samples_10 = num_train_data // 100
samples_100 = num_train_data // 100

results = {}
# for clf in [clf_benchmark_1, clf_benchmark_2, clf_benchmark_3]:
for clf in [clf_benchmark_1]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1]):
        print(i)
        results[clf_name][i] = train_validate(clf, samples, X_train, y_train, X_valid, y_valid)
    


# In[10]:


print(results)

