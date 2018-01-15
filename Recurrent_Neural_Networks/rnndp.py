# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the training set
dataset_training = pd.read_csv('Google_Stock_Price_Train.csv')
dataset_training = dataset_training.iloc[:,1:2].values

#Feature Scaling using Normalisation using MinMaxScaler. MinMaxScaler has the capability to reverse the values when needed
from sklearn.preprocessing import StandardScaler,MinMaxScaler
sc = MinMaxScaler()
dataset_training = sc.fit_transform(dataset_training)

#Getting the inputs and outputs
X_train=dataset_training[0:1257]
y_train=dataset_training[1:1258] 

#reshaping
X_train = np.reshape(X_train,(1257,1,1))

#Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#Initialising the RNN
regressor = Sequential()

#Adding the input layer and LSTM layer
regressor.add(LSTM(units = 1,activation = 'sigmoid',input_shape=(None,1)))

#Adding the output layer
regressor.add(Dense(units = 1))

#for RNN rmsprop optimizer is most used but takes more memory. Loss is mean_squared_error as we are checking regression
regressor.compile(optimizer='adam',loss='mean_squared_error')

#Fitting the model
regressor.fit(X_train,y_train,batch_size = 32 ,epochs = 200 )

#Making the predictions and visualising the resuls

#getting the google stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

#getting the predicted stock price
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs,(20,1,1))

predicted_stock_price = regressor.predict(inputs) 
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualising the results
plt.plot(real_stock_price,color = 'red',label = 'Real Google stock price')
plt.plot(predicted_stock_price,color = 'blue',label = 'Predicted Google stock price')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Google stock price')
plt.legend()
plt.show()

#Evalualating the RNN
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price,predicted_stock_price))



#Tuning the RNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_training = pd.read_csv('Google_Stock_Price_Train.csv')
dataset_training = dataset_training.iloc[:,1:2].values

from sklearn.preprocessing import StandardScaler,MinMaxScaler
sc = MinMaxScaler()
dataset_training = sc.fit_transform(dataset_training)

X_train = []
y_train = []
for i in range(20,1258):
    X_train.append(dataset_training[i-20:i,0])
    y_train.append(dataset_training[i,0])
X_train,y_train = np.array(X_train),np.array(y_train) 

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
#Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

regressor = Sequential()

regressor.add(LSTM(units = 3,activation = 'sigmoid',input_shape=(None,1)))

regressor.add(Dense(units = 1))

regressor.compile(optimizer='adam',loss='mean_squared_error')

regressor.fit(X_train,y_train,batch_size = 32 ,epochs = 200 )

test_set = pd.read_csv('Google_Stock_Price_Test.csv')
test_set = dataset_test.iloc[:,1:2].values
real_stock_price = np.concatenate((dataset_training[0:1258], test_set), axis = 0)

scaled_stock_price = real_stock_price
scaled_stock_price = sc.fit_transform(scaled_stock_price)
inputs = []
for i in range(1258, 1278):
    inputs.append(scaled_stock_price[i-20:i, 0])
inputs = np.array(inputs)
inputs = np.reshape(inputs,(inputs.shape[0],inputs.shape[1],1))

predicted_stock_price = regressor.predict(inputs) 
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price,color = 'red',label = 'Real Google stock price')
plt.plot(predicted_stock_price,color = 'blue',label = 'Predicted Google stock price')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Google stock price')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price,predicted_stock_price))