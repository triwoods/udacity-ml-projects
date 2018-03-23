from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation

def linear_regression(dim_in, dim_out):
    """
    Linear regression model
    dim_in: input dimension
    dim_out: output dimension
    """
    model = Sequential()
    model.add(Dense(dim_out, input_shape=(dim_in,), activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model

def mlp(dim_in, dim_out, nl=1, nn=128, dropout=False):
    """
    Multilayer perceptron network
    dim_in: input dimension
    dim_out: output dimension
    nl: number of hidden layers
    nn: number of neurons in each hidden layer
    dropout: whether to use dropout in hidden layer
    """
    model = Sequential()
    model.add(Dense(nn, input_shape=(dim_in,), activation='relu'))
    for i in range(nl-1):
        model.add(Dense(nn, activation='relu'))
        if dropout:
            model.add(Dropout(0.5))
    model.add(Dense(dim_out))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model

def lstm(dim_in, dim_out, time_step, nl=1, nn=32):
    """
    Long short-term memory network
    dim_in: input dimension
    dim_out: output dimension
    time_step: number of time step
    nl: number of stacks of LSTM layer
    nn: number of LSTM unit in each LSTM stack
    """
    model = Sequential()
    model.add(LSTM(nn, return_sequences=bool(nl-1), input_shape=(time_step, dim_in)))
    if nl > 1:
        for nl in range(nl-2):
            model.add(LSTM(nn, return_sequences=True))
        model.add(LSTM(nn, return_sequences=False))
    model.add(Dense(dim_out))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model
