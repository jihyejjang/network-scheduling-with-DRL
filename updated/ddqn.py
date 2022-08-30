from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import *
import tensorflow as tf
from utils import *

tf.compat.v1.disable_eager_execution()

def create_model(lr):
    model = Sequential()
    model.add(Dense(128, input_dim=INPUT_SIZE, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=ALPHA))
    # model.add(Dropout(DROPOUT))
    model.add(Dense(128, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=ALPHA))
    model.add(Dense(OUTPUT_SIZE, activation='linear', kernel_initializer='he_normal'))  # relu
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))
    return model


class DoubleDeepQNetwork:
    def __init__(self,args):
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        self.loss_history = []

        self.model = create_model(args.learningrate)
        self.target_model = create_model(args.learningrate)
        # else:
        #     self.model = tf.keras.models.load_model(WEIGHT_FILE)
        #     self.target_model = tf.keras.models.load_model(WEIGHT_FILE)

        self.update_target_model()

    def train(self, x, y, sample_weight=None, epochs=1, verbose=0):  
        loss = []
        history = self.model.fit(x, y, batch_size=len(x), sample_weight=sample_weight, epochs=epochs, verbose=verbose)
        loss.append(history.history['loss'][0])  
        return min(loss)

    def test(self, weight_file):
        self.model.load_weights(weight_file)

    def predict_one(self, state, target=False):
        return self.predict(np.array(state).reshape(1, INPUT_SIZE), target=target).flatten()

    def predict(self, state, target=False):
        if target:  # get prediction from target network
            return self.target_model.predict(state)
        else:  # get prediction from local network
            return self.model.predict(state)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, filename):
        self.model.save(filename)
