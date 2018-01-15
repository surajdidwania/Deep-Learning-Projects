# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values  #Is helped to create numpy array
y = dataset.iloc[:,-1].values

#Festure Scaling
from sklearn.preprocessing import StandardScaler,MinMaxScaler
sc= MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

#Implementation of Self organising maps
from minisom import MiniSom
som = MiniSom(10,10,15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X,100)

#Visualiisng results
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,markers[y[i]],markeredgecolor = colors[y[i]],markerfacecolor = 'None',markersize = 10,markeredgewidth = 2)
show()
    
#Finding the frauds

mappings = som.win_map(X)
fraud = np.concatenate((mappings[(7,0)],mappings[(8,1)]),axis=0)
fraud = sc.inverse_transform(fraud)

#GOING FROM UNSUPERVISED LEARNING TO SUPERVISED LEARNING

#Creating the matrix of features
customers = dataset.iloc[:,1:].values

#Creating the depedent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in fraud:  
        is_fraud[i]=1


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


#importing keras Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#Initialising ANN
classifier = Sequential()

#Adding the input layer and hidden layer with dropout p not greater than 0.5
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=15))
classifier.add(Dropout(p=0.1))

#Adding the second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
classifier.add(Dropout(p=0.1))

#Adding the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN into training set
classifier.fit(customers,is_fraud,batch_size = 1,epochs=5)
#Predicting the fucntion
y_pred = classifier.predict(customers)

y_pred = np.concatenate((dataset.iloc[:,0:1].values,y_pred),axis=1)

y_pred = y_pred[y_pred[:,1].argsort()]
