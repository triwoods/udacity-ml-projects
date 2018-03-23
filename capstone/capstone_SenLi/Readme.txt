The project is using the following python libs:

- numpy
- scipy
- matplotlib
- keras
- tensorflow

models.py - contains models built with keras

utils.py - contains utility functions for data processing


./data
The feature extracted data for training and testing in the capstone project can be found in /data

The feature is line spectrum frequency (LSF), to save space, the value is store as 'int16' instead of 'float32' using Q15 format

The range of LSF value is (0, pi/2)

To convert the Q15 format to the LSF physical value, the following equation can be used
"""
Q_value = 15
y = x.astype('float') * np.pi / np.power(2, Q_value)
"""

You can navigate to the python notebook to start running the code and have fun.