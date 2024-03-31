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

#Try testing the error of AI/Testing primes
def is_prime(num):
    if num <=  1:
        return False
    if num == 2: 
        return True
    if num % 2 == 0:
        return False 
    max_devisor = int(np.sqrt(num))
    for i in range(3, max_devisor + 1, 2):
        if num % i == 0:
            return False
    return True

##################################################################################################
#Could make a csv where one side is numbers to 10 000 
#While the other is 0 or 1 
#This will allow binary crossentropy
##################################################################################################

#for Primes generated locally 
# numbers = np.arange(1, 10001)
# labels = np.array([1 if is_prime(num) else 0 for num in numbers])

#for document reading 
document = pd.read_csv('primenumbers.csv')
numbers = document['Primes'].values
numbers = numbers.reshape(-1, 1)
labels = np.array([1 if is_prime(num) else 0 for num in numbers])

#Build and train model 
model = tf.keras.models.Sequential()

#data shape
model.add(tf.keras.Input(shape=numbers.shape[1])) 

#ReLu = Linear regression model 
for i in range(0, 14):
    model.add(tf.keras.layers.Dense( 2 ** (i), activation='relu'))

#Tanh - -1 to 1 
for i in range(0,7):
    model.add(tf.keras.layers.Dense( 2 ** (i), activation='tanh'))

#Binary 0 and 1
model.add(tf.keras.layers.Dense( 1, activation='sigmoid'))

#Compile/Build model:
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy']) 

#Printing function to parse into model
class PrintSequenceCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1 == 0:  # Adjust the frequency of printing as needed
            predictions = np.round(self.model.predict(numbers)).flatten()
            print(f"Epoch {epoch+1} Predictions: {numbers[predictions == 1].flatten()}")
            print(f"labels: {labels}")

#Train model
model.fit(numbers, labels, epochs=10, validation_split=0.2, callbacks=[PrintSequenceCallback()])

#Compares AI prediction and actual data (Or something)
test_loss, test_acc = model.evaluate(numbers, labels)

#Just testing parameters
print('Test Accuracy:', test_acc)
print(f"{numbers}")
print(f"{labels}")