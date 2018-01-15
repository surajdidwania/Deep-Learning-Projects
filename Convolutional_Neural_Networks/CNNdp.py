# -*- coding: utf-8 -*-

#Import libraries
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Data Preprocessing - already done as the tyraing set(8000 images) and test set(2000 images) has been prepared with.

#Part 1 Building the CNN

#Import keras Packages and libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#Initialising CNN as sequence of layers
classifier= Sequential()

#!: Convolution
input_size = (64,64)
classifier.add(Convolution2D(32,3,3,input_shape = (*input_size,3),activation='relu')) #32 feature detector of 3*3 matrix will 
#get 32 feature maps with images in 3d Array with 3 layers and 64*64 images because of tensorflow backend

# 2: Pooling: Pooled feature maps
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(32,3,3,activation='relu')) #to improve accuracy to input pooled feature maps 
classifier.add(MaxPooling2D(pool_size = (2,2)))

#3: flattening
classifier.add(Flatten())

#$: Connection
classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 1, activation='sigmoid'))

classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy']) 
#loss = 'categorical_crossentropy'

#Part 2: Fitting the model into images
#Image AUgmentatation and how dows it prevent overfitting
#Image Augmentation is the technique which will only enrich our training set w/o adding images allow giving good result with little or no overfitting 
#it also allows rotating of images
#Keras documnetation Image preprocessing and helps in fitting images in CNN and perform IA
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
batch_size =32
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=input_size,
                                                 batch_size=batch_size,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                        target_size= input_size,
                                                        batch_size=batch_size,
                                                        class_mode='binary')

classifier.fit_generator(
                        training_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000)


# Making single predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64, 64))
test_image=image.img_to_array(test_image)
test_image =np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
training_set.class_indices

#Tuning of CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def built_classifier(optimizer):
    classifier= Sequential()
    classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3),activation='relu'))  
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Convolution2D(32,3,3,activation='relu'))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 128, activation='relu'))
    classifier.add(Dense(output_dim = 1, activation='sigmoid'))
    classifier.compile(optimizer = optimizer,loss = 'binary_crossentropy',metrics=['accuracy']) 
    
classifier = KerasClassifier(build_fn = built_classifier)
parameters = {'batch_size':[25,50],'optimizer' :['adam','rmsprop'],'epochs':[100,500]}
grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy',cv=10)
grid_search.fit_generator(
                        training_set,
                        steps_per_epoch=8000,
                        validation_data=test_set,
                        validation_steps=2000)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_