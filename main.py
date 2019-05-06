import numpy as np 
from enum import Enum
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tkinter import messagebox
import app
from app import input_vector
	
def app():
    from app import line_drawing

# General framework configuration class
class Options:
    def __init__(self):
        self.input_size = 784
        self.target_size = 10
        self.hidden_layer_size = [512, 256]

# FrameWork base class
class FrameWork:
    def __init__(self, options):
        (self.train_input, self.train_target), (test_input, test_target) = tf.keras.datasets.mnist.load_data()
        self.options = options
        self.train_input = self.train_input.reshape(-1, self.options.input_size) / 255.0
        self.tf_model_path = 'tf_model.h5';
    def set_training_data(self, input_vector, label):
        self.train_input = np.concatenate((self.train_input, input_vector))
        self.train_target = np.append(self.train_target, label)

# TensorFlow sub class
class TensorFlow(FrameWork):
    def load_model(self):
	    self.model = keras.models.load_model(self.tf_model_path)
    def create_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(self.options.hidden_layer_size[0], activation=tf.nn.relu, input_shape=(self.options.input_size,)))
        num = 1
        while len(self.options.hidden_layer_size) > num:
            self.model.add(tf.keras.layers.Dense(self.options.hidden_layer_size[1], activation=tf.nn.relu))
            num+=1
        self.model.add(tf.keras.layers.Dense(self.options.target_size,activation=tf.nn.softmax))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy'])
    def run_model(self, input_vector):
        prediction = self.model.predict(input_vector)
        return prediction
    def train_model(self, epoch):
        try:
            self.model.fit(self.train_input, self.train_target, epochs=epoch)
        except:
            print("Model not loaded")
    def save_model(self):
        try:
            self.model.save(self.tf_model_path)
        except:
            print("Model not loaded")

# PyTorch sub class (to be added)
# class PyTorch(FrameWork):
    # def load_model(self):
    # def create_model(self):
    # def run_model(self, input_vector):
    # def train_model(self, epoch):
    # def save_model(self):

	
if __name__ == "__app__":
    app()

# load model
frame_work = TensorFlow(Options())
frame_work.load_model()
# predict 
prediction = frame_work.run_model(input_vector)
messagebox.showinfo("Prediction is: ", np.argmax(prediction))
# update training set
frame_work.set_training_data(input_vector, np.argmax(prediction))
# retrain model
frame_work.train_model(3)
frame_work.save_model()
