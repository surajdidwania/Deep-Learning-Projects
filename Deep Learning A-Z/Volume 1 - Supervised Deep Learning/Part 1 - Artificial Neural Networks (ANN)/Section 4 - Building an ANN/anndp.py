# -*- coding: utf-8 -*-


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing csv file
dataset = pd.read_csv('Churn_Modelling.csv')   #Dataframes
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#Splitting the dataset onto Training set and Test Set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing keras Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#Initialising ANN
classifier = Sequential()

#Adding the input layer and hidden layer with dropout p not greater than 0.5
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
classifier.add(Dropout(p=0.1))

#Adding the second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
classifier.add(Dropout(p=0.1))

#Adding the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN into training set
classifier.fit(X_train,Y_train,batch_size = 10,epochs=100)

#Predicting the fucntion
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)


#Predicting the single observations
new_pred = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred = (new_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)

#Evalulating and imporving the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#Build the architecture of ANN
def built_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = built_classifier,batch_size = 10,epochs=100)
accuracies = cross_val_score(estimator = classifier,X = X_train, y = Y_train,cv=10,n_jobs = -1)
mean = accuracies.mean()
var = accuracies.std()

#Tuning the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

#Build the architecture of ANN
def built_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = built_classifier)
parameters = {'batch_size':[25,32],'epochs':[100,500],'optimizer': ['adam','rmsprop']}
grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy',cv=10)
grid_search = grid_search.fit(X_train,Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


