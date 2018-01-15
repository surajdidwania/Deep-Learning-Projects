# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#reading of data
movies = pd.read_csv('ml-1m/movies.dat',sep = '::' ,header = None, engine = 'python',encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat',sep = '::' ,header = None, engine = 'python',encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat',sep = '::' ,header = None, engine = 'python',encoding = 'latin-1')

#Preparing the trainign set and test set
training_set = pd.read_csv('ml-100k/u1.base',delimiter = '\t')
training_set = np.array(training_set,dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test',delimiter = '\t')
test_set = np.array(test_set,dtype = 'int') 

def gettingusermovie():
    nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
    nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))
    return nb_users,nb_movies

nb_users,nb_movies = gettingusermovie()

def convert_data(data):
    new_data = []
    for id_users in range(1,nb_users+1):
        id_ratings = data[:,2][data[:,0]==id_users]
        id_movies = data[:,1][data[:,0]==id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data
    
training_set = convert_data(training_set)
test_set = convert_data(test_set)


#Converting the data into torch sensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#Creating the architecture of neural network
class SAE(nn.Module): #Inheritance class take everything from parent class
    def __init__(self, ):
        super(SAE,self).__init__()  #Self refers to object
        self.fc1 = nn.Linear(nb_movies , 20)   #no of feature in the input , # no of neurons in hidden layer, features in the hidden layer detects some feature in unsupervised lesarning that auto encode detects
        self.fc2 = nn.Linear(20,10)
        self.fc3 = nn.Linear(10,20) #Encoding done
        self.fc4 = nn.Linear(20,nb_movies) # Now specifify the activation fucntion
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
        
sae = SAE()
criterion = nn.MSELoss()
optimiser = optim.RMSprop(sae.parameters(),lr = 0.01,weight_decay = 0.5)

#Training the SAE
nb_epochs = 200
for epochs in range(1,nb_epochs+1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0)>0:
            output = sae.forward(input)
            #optimising the input and memory
            target.require_grad = False
            output[target==0] = 0
            loss=criterion(output,target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0)+1e-10)
            loss.backward()
            train_loss+= np.sqrt(loss.data[0]*mean_corrector)
            s += 1
            optimiser.step()
    print('epoch:' + str(epoch) + 'loss:' + str(train_loss/s))
            
    
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0)>0:
        output = sae.forward(input)
    #optimising the input and memory
        target.require_grad = False
        output[target==0] = 0
        loss=criterion(output,target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0)+1e-10)
        test_loss+= np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('loss:' + str(test_loss/s))













