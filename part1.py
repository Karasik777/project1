#Initialisation 
import math
import numpy
import matplotlib
import seaborn

import keras
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('dataset.csv')

x = dataset.drop(columns=["diagnosis(1=m, 0=b)"]) #Column of file

y = dataset["diagnosis(1=m, 0=b)"] 

#Split data into testing and training set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10000)

#Compares AI prediction and actual data (Or something)
model.evaluate(x_test, y_test)







