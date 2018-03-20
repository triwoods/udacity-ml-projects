The feature extracted data for training and testing in the capstone project can be found in /data
The feature is line spectrum frequency (LSF), to save space, the value is store as 'int16' instead of 'float32' using Q15 format
The range of LSF value is [0, pi]
To convert the Q15 format to the LSF physical value, the following equation can be used
"""
Q_value = 15
y = x.astype('float') * 2 / np.power(2, Q_value)
"""
