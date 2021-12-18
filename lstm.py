import math
import matplotlib.pyplot as plt
from tensorflow.python import tf2
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


def load_data(path):
    return pd.read_csv(path, parse_dates=[0])

path = r'E:/University/COVID-19/coronavirus-data-master/trends/data-by-day.csv'
df = load_data(path)
# i_tuple = (20, 21, 22)
predictions = 90
training_case_count = df.iloc[:534, 1:2].values
test_scase_count = df.iloc[534:, 1:2].values

scaler = MinMaxScaler(feature_range=(0, 1))
case_count_scaled = scaler.fit_transform(training_case_count)
X_train = []
y_train = []
for i in range(60, 534):
    X_train.append(training_case_count[i-60:i, 0])
    y_train.append(training_case_count[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units=50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units=1))
# Compiling the RNN
model.compile(optimizer='adam', loss='mean_squared_error')
# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs=100, batch_size=32)

# dataset_train = df.iloc[:534, 1:2]
# dataset_test = df.iloc[534:, 1:2]
# dataset_total = pd.concat((dataset_train, dataset_test), axis=0)
# inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# inputs = inputs.reshape(-1,1)
# inputs = scaler.transform(inputs)
# X_test = []
# for i in range(60, 519):
#     X_test.append(inputs[i-60:i, 0])
# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# print(X_test.shape)
# (459, 60, 1)


# Visualising the results


