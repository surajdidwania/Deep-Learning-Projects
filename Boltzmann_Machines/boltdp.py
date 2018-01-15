# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

movies = pd.read_csv('ml-1m/movies.dat',sep = '::',header=None,engine='python',encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat',sep = '::',header=None,engine='python',encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat',sep = '::',header=None,engine='python',encoding = 'latin-1')

#Preparing the training set and Test Set
training_set = pd.read_csv('ml-100k/u1.base',delimiter='\t') #Pandas craeet a pandas
training_set = np.array(training_set,dtype = 'int') # np.array convert dataframe into array required by python
test_set = pd.read_csv('ml-100k/u1.test',delimiter='\t') #Pandas craeet a pandas
test_set = np.array(test_set,dtype = 'int') 

#getting the users and Movies #Get total no of user and total no of movie in test set or training set bcoz its is randomly separated and has top be splitted accordinlgy
#Total no of movies and total users
def gettingusermovie():
    nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
    nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))
    return nb_users,nb_movies
    
nb_users,nb_movies = gettingusermovie()

def convert(data):
    new_data = [] #Initialise a list
    for id_users in range(1,nb_users+1):
        id_movies = data[:,1][data[:,0]==id_users]
        id_ratings = data[:,2][data[:,0]==id_users]
        ratings  = np.zeros(nb_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data
        
training_set = convert(training_set)
test_set = convert(test_set)

#Converting the data into torch tensors #It stores the element of same data type
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#converting the ratings into binary ratings (1 liked) (0 Disliked)
training_set[training_set==0] = -1 # For all the 0 ratings 
training_set[training_set==1] = 0
training_set[training_set==2] = 0
training_set[training_set>=3] = 1 

test_set[test_set==0] = -1 # For all the 0 ratings 
test_set[test_set==1] = 0
test_set[test_set==2] = 0
test_set[test_set>=3] = 1

#Creating the architecture of neural network
class RBM():
    def __init__(self,nv,nh):
        self.Weight = torch.randn(nh,nv)
        self.a= torch.randn(1,nh) #Bias for hidden nodes
        self.b = torch.randn(1,nv) #bias for Visible nodes
     
    def sample_h(self,x): #x corresponds to visible neurons given viisble nodes
        wx = torch.mm(x,self.Weight.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self,y): #x corresponds to visible neurons given viisble nodes
        wy = torch.mm(y,self.Weight)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
        
    #Contrasitive Divergence
    def train(self,v0,vk,ph0,phk):
        self.Weight += torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)
        self.b += torch.sum((v0-vk),0)
        self.a += torch.sum((ph0-phk),0)
        
nv = len(training_set[0])
nh = 100 #Can be tuned
batch_size = 100 #Can be tunable
rbm = RBM(nv,nh) # we will update the weights after several observations, Batches will have same no of observations

#Training RBM
nb_epoch = 10
for epoch in range(1,nb_epoch+1):
    train_loss = 0 #In beginning the loss achieved is 0
    s = 0. # s has type float, S for normalisation
    for id_user in range(0,nb_users - batch_size , batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_=rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[vk>=0]))
        s += 1.
    print('epoch: '+str(epoch) + 'loss: '+ str(train_loss/s))
    
#testing thr rbm
test_loss = 0 #In beginning the loss achieved is 0
s = 0. # s has type float, S for normalisation
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0])>0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
        s += 1.
print('epoch: '+str(epoch) + 'loss: '+ str(test_loss/s))

    
            
        
    