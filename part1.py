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
x = dataset.drop(columns=["2"])
y = dataset["2"]

#Need to filter data here like statistics
#Primes is prediction 
#Cancer is taking in values and spitting out 1 or 0 


#Split data into testing and training set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Build and train model 
model = tf.keras.models.Sequential()

#NOT WORKING CODE
#model.add(tf.keras.layers.Dense(512, input_shape=x_train.shape, activation='sigmoid'))

model.add(tf.keras.Input(shape=(x_train.shape[1]))) 
model.add(tf.keras.Input(shape=(x_test.shape[1]))) 

model.add(tf.keras.Input(shape=(y_train.shape))) 
model.add(tf.keras.Input(shape=(y_test.shape)))

#Some parameters
network_num = 12
attempts = 10000000

for i in range(0, network_num):
    model.add(tf.keras.layers.Dense( 2 ** (network_num - i), activation='sigmoid'))


#Lower error than before
#model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['accuracy'])

#Reasonable error due to wrong generation
#need to set up machine learning for prediction instead of training
#model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.compile(optimizer='Adadelta', loss='mean_squared_error', metrics=['accuracy']) #Very interesting result
model.fit(x_train, y_train, epochs=attempts)

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






