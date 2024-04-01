#Initialisation 
import numpy as np
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

#Runs model
def run_model(numbers, labels):
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
    #Build and run:
    model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['accuracy']) 
    #Printing function to parse into model
    class PrintSequenceCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 1 == 0:  # Adjust the frequency of printing
                predictions = np.round(self.model.predict(numbers)).flatten()
                print(f"Epoch {epoch+1} Predictions: {numbers[predictions == 1].flatten()}")
                print(f"labels: {labels}")
    #Train model
    # model.fit(numbers, labels, epochs=10, validation_split=0.2, callbacks=[PrintSequenceCallback()])            
    model.fit(numbers, labels, epochs=15, validation_split=0.2)
    test_loss, test_acc = model.evaluate(numbers, labels)
    return model

#Runs script, saves sucessful models
def main():
    #Generate numbers  
    numbers = np.arange(1, 7920)
    #Data cleaning
    numbers = numbers.reshape(-1, 1)
    #Label primes
    labels = np.array([1 if is_prime(num) else 0 for num in numbers])

    count = 0 #Sucessful models count
    while True:
        model = run_model(numbers, labels)
        #Make prediction
        prediction = model.predict(labels)
        prediction = [0 if num < 0.5 else 1 for num in prediction]
        #Evaluate accuracy of prediction
        accuracy_of_prediction = accuracy_score(labels, prediction)
        print(f"Prediction Accuracy: {accuracy_of_prediction}")
        if accuracy_of_prediction >= 0.98:
            model.save(f"Good_model_{count + 1}")
            count = count + 1 
        elif count == 5:
            break

#Run it
if __name__ == "__main__":
    main()
