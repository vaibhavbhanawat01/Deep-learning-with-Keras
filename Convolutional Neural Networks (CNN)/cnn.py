# -*- coding: utf-8 -*-
"""
Created on Sat May  9 22:05:52 2020

@author: vaibhav_bhanawat
"""

# Building the CNN.

# importing the keras libaries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Initialising the CNN
classifier = Sequential()

# Convolution
classifier.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(64, 64, 3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Convolution 2nd layer
classifier.add(Convolution2D(32, 3, 3, border_mode='same', activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

## Full connection- making classic ANN
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the Model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Preprocessing phase to reduce the overfitting and Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_generator = train_datagen.flow_from_directory('dataset/dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_generator = test_datagen.flow_from_directory('dataset/dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')
classifier.fit_generator(training_generator,
                    steps_per_epoch = 8000,
                    epochs =3,
                    validation_data=test_generator,
                    validation_steps =2000)
