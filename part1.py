#Initialisation 
import sys
import math
import numpy as np
import matplotlib
import seaborn

import keras
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

#for cancer dataset
#dataset = pd.read_csv('dataset.csv')
# x = dataset.drop(columns=["diagnosis(1=m, 0=b)"]) #Column of file
# y = dataset["diagnosis(1=m, 0=b)"] 

#for primes dataset:
# dataset = pd.read_csv('primenumbers.csv')
# x = dataset.drop(columns=["2"])
# y = dataset["2"]

#Try testing the error of AI
def is_prime(num):
    flag = False
    if num > 1:
        for i in range(2,num):
            if num % i == 0:
                flag = True
                break

        return flag

#for generated primes:
numbers = np.arange(1, 10001)
labels = np.array([1 if is_prime(num) else 0 for num in numbers])

#for document reading 
# document = pd.read_csv('primenumbers.csv')
# numbers = document['2'].values
# labels = np.array([1 if is_prime(num) else 0 for num in numbers])

#Need to filter data here like statistics
#Primes is prediction 
#Cancer is taking in values and spitting out 1 or 0 

#Build and train model 
model = tf.keras.models.Sequential()

#data shape
model.add(tf.keras.Input(shape=1,)) 

#Adding layers
model.add(tf.keras.layers.Dense( 256, activation='relu'))
model.add(tf.keras.layers.Dense( 128, activation='relu'))
model.add(tf.keras.layers.Dense( 64, activation='relu'))
model.add(tf.keras.layers.Dense( 1, activation='sigmoid'))


#Reasonable error due to wrong generation
#need to set up machine learning for prediction instead of training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #Very interesting result
model.fit(numbers, labels, epochs=50, validation_split=0.2)

#Compares AI prediction and actual data (Or something)
#model.evaluate(x_test, y_test)

test_loss, test_acc = model.evaluate(numbers, labels)
print('Test Accuracy:', test_acc)









