# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Festure Scaling
from sklearn.preprocessing import StandardScaler,MinMaxScaler
sc= MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

#Implementation of Self organising maps
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
fraud = np.concatenate((mappings[(7,7)],mappings[(6,6)]),axis=0)
fraud = sc.inverse_transform(fraud)