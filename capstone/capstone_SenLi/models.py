# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor

from keras.models import Sequential
from keras.layers.core import Dense, Activation

def linear_regression():
    model = Sequential()
    model.add(Dense(6, input_shape=(10,), activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model


def mlp():
    model = Sequential()
    model.add(Dense(128, input_shape=(10,), activation='relu'))
    model.add(Dense(6))
    model.compile(loss='mse', optimizer='adam')
    return model
