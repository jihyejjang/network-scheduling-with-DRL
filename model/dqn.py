#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import *
import numpy as np

class DQN: 
    def __init__(self,queue,cycle): 
        self.input = queue #agent는 queue 1개를 담당하지만 모든 queue의 state를 고려, 해당 큐에 몇 개의 flow가 대기중인지? or 몇 byte?
        self.output = 2**cycle #available action, 2^몇 개의 slot을 한 사이클로 정할것인지(cycle)
        self.learning_rate = 0.0001
        self.loss_history = []
        self.model = self.create_model() #현재 state에 대한 model
        self.target_model = self.create_model() #next state에 대한 model

    # create the neural network to train the q function 
    def create_model(self): 
        model = Sequential()
        model.add(Dense(24, input_dim= self.input, activation= 'relu')) 
        model.add(Dense(48, activation= 'relu'))
        model.add(Dense(24, activation= 'relu'))
        model.add(Dense(self.output)) #allowed action 
        model.compile(loss= 'mean_squared_error', optimizer= Adam(lr= self.learning_rate)) #optimizer의 learning rate 주의
        return model 
    
    def train(self, x, y, sample_weight=None, epochs=1, verbose=0):  # x is the input to the network and y is the output
        loss=[]
        history=self.model.fit(x, y, batch_size=len(x), sample_weight=sample_weight, epochs=epochs, verbose=verbose)
        loss.append(history.history['loss'][0]) # loss 기록
        return min(loss)
        
    def test(self,weight_file):
        self.model.load_weights(weight_file)
    
    def predict_one(self, state, target=False):
        return self.predict(np.array(state).reshape(1,self.input), target=target).flatten()

    def predict(self, state, target=False):
        if target:  # get prediction from target network
            return self.target_model.predict(state)
        else:  # get prediction from local network
            return self.model.predict(state)
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # save our model 
    def save_model(self, filename):
        self.model.save(filename)

