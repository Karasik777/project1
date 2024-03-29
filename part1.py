#Initialisation 
import sys
import math
import numpy
import matplotlib
import seaborn

import keras
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_predict




#for cancer dataset
#dataset = pd.read_csv('dataset.csv')
# x = dataset.drop(columns=["diagnosis(1=m, 0=b)"]) #Column of file
# y = dataset["diagnosis(1=m, 0=b)"] 


#for primes dataset
dataset = pd.read_csv('primenumbers.csv')
x = dataset.drop(columns="2")
y = dataset["2"]




#Split data into testing and training set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


#Build and train model 
model = tf.keras.models.Sequential()

#NOT WORKING CODE
# model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape, activation='sigmoid'))
# model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.add(tf.keras.Input(shape=(x_train.shape[1],))) 
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#Compilation 
#Model only for cancer
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Lower error than before
#model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['accuracy'])

#Reasonable error due to wrong generation
#need to set up machine learning for prediction instead of training
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100)

#Compares AI prediction and actual data (Or something)
model.evaluate(x_test, y_test)



#Try testing the error of AI
def is_prime(num):
    flag = False
    if num > 1:
        for i in range(2,num):
            if num % i == 0:
                flag = True
                break

        return flag






