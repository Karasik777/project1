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
from sklearn.metrics import accuracy_score


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


#for Primes generated locally 
numbers = np.arange(1, 7920)

#for document reading 
# document = pd.read_csv('primenumbers.csv') #This document only contains primes
# numbers = document['Primes'].values


#Data pre-processing: 
numbers = numbers.reshape(-1, 1)
#label primes
labels = np.array([1 if is_prime(num) else 0 for num in numbers])

#Build and train model 
model = tf.keras.models.Sequential()

#Data Shape
model.add(tf.keras.Input(shape=numbers.shape[1])) 

#ReLu = Rectified Linear model
for i in range(0, 6):
    model.add(tf.keras.layers.Dense( 2 ** (i), activation='relu'))

model.add(tf.keras.layers.Dense( 8, activation='tanh'))

#Binary 0 and 1
# model.add(tf.keras.layers.Dense( 2, activation='sigmoid'))
model.add(tf.keras.layers.Dense( 1, activation='sigmoid'))




#Compile/Build model:
#Good optimisers: 'sgd', 'Adadelta', 'Adam', 'adam'
#Losses: 'mean_squared_error', 'binary_crossentropy'
#Metrics: 'accuracy' always
#Batch_size: could try 32
model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['accuracy']) 

#Printing function to parse into model
class PrintSequenceCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1 == 0:  # Adjust the frequency of printing
            predictions = np.round(self.model.predict(numbers)).flatten()
            print(f"Epoch {epoch+1} Predictions: {numbers[predictions == 1].flatten()}")
            print(f"labels: {labels}")

#Train model
#This will print the results as it trains
# model.fit(numbers, labels, epochs=10, validation_split=0.2, callbacks=[PrintSequenceCallback()]) 
model.fit(numbers, labels, epochs=10, validation_split=0.2)

#Compares AI prediction and actual data (Or something)
test_loss, test_acc = model.evaluate(numbers, labels)

#Make prediction
prediction = model.predict(labels)
prediction = [0 if num < 0.5 else 1 for num in prediction]

#Evaluate accuracy of prediction
accuracy_of_prediction = accuracy_score(labels, prediction)

#Just testing parameters
print(f"Labels: {labels}")
print(f"Prediction: {prediction}")
print(f"Prediction Accuracy: {accuracy_of_prediction}")


#This code allows to save the current model:
#model.save('New_Model_Attempt')

#Deletes model in this file
#del model

#Reloading model: 
#model = load_model('MODEL_NAME')



#Write a function to run the models inifnitely untill a good one is reached 
#Like while 1 run if model accuracy above blah save it 