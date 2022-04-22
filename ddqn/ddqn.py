#!/usr/bin/env python

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import *
import tensorflow as tf
from parameter import *

tf.compat.v1.disable_eager_execution()


def create_model():
    model = Sequential()
    model.add(Dense(128, input_dim=INPUT_SIZE, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=ALPHA))
    # model.add(Dropout(DROPOUT))
    model.add(Dense(128, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=ALPHA))
    model.add(Dense(OUTPUT_SIZE, activation='linear', kernel_initializer='he_normal'))  # relu
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=LEARNING_RATE))
    return model


class DoubleDeepQNetwork:
    def __init__(self):
        self.loss_history = []

        if FIRST_TRAIN :
            self.model = create_model()
            self.target_model = create_model()
        else:
            self.model = tf.keras.models.load_model(WEIGHT_FILE)
            self.target_model = tf.keras.models.load_model(WEIGHT_FILE)

        self.update_target_model()
    # create the neural network to train the q function

    def train(self, x, y, sample_weight=None, epochs=1, verbose=0):  # x is the input to the network and y is the output
        loss = []
        history = self.model.fit(x, y, batch_size=len(x), sample_weight=sample_weight, epochs=epochs, verbose=verbose)
        loss.append(history.history['loss'][0])  # loss 기록
        return min(loss)

    def test(self, weight_file):
        self.model.load_weights(weight_file)

    def predict_one(self, state, target=False):
        return self.predict(np.array(state).reshape(1, INPUT_SIZE), target=target).flatten()

    def predict(self, state, target=False):
        if target:  # get prediction from target network
            return self.target_model.predict(state)
        else:  # get prediction from local network
            # print (state)
            return self.model.predict(state)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # save our model 
    def save_model(self, filename):
        self.model.save(filename)
