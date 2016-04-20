"""
Recursive neural net using fancy LSTM and Keras
"""

from sklearn import base
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape, Permute
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, AveragePooling1D, MaxPooling1D
from keras.layers.wrappers import TimeDistributed


class RNN(base.BaseEstimator, base.ClassifierMixin):

  def __init__(self, dropout=0.5, epochs=15):
    self.dropout = dropout
    self.epochs = epochs

  def __getstate__(self):
    self.model_.save_weights(self.filename, overwrite=True)
    state = dict(self.__dict__)
    del state['model_']
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self.model_ = self._create_model(self.shape_)
    self.model_.load_weights(self.filename)

  def fit(self, X, validation_data=None, samples=0):
    #self.shape_ = X.shape
    self.shape_ = (1,2000,120)
    self.model_ = self._create_model(self.shape_)
    self.model_.fit_generator(X, samples, self.epochs, validation_data=validation_data)

  def predict(self, X):
    return self.model_.predict_classes(X)[0]

  def set_filename(self, filename):
    self.filename = filename

  def _create_model(self, shape):
    print(shape)
    return self._conv_model(shape)

  def _conv_model(self, shape):
    filter_width = 8
    n_filters = 512
    timesteps = shape[1]
    features = shape[2]
    model = Sequential()
    model.add(Convolution1D(n_filters*2, filter_width/2, activation='relu', input_shape=(timesteps,features)))
    timesteps -= filter_width/2 - 1
    model.add(MaxPooling1D(filter_width/2))
    timesteps /= filter_width / 2
    model.add(Dropout(self.dropout))
    model.add(Convolution1D(n_filters, filter_width, activation='relu'))
    timesteps -= filter_width - 1
    model.add(MaxPooling1D(filter_width))
    timesteps /= filter_width
    model.add(Dropout(self.dropout))
    model.add(Convolution1D(n_filters, filter_width, activation='relu'))
    timesteps -= filter_width - 1
    model.add(MaxPooling1D(timesteps))
    model.add(Dropout(self.dropout))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(15))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    from keras.utils.visualize_util import plot
    plot(model, to_file='model.png', show_shapes=True)
    return model

  def _lstm_model(self, shape):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(shape[1],shape[2])))
    model.add(Dropout(0.5))
    model.add(AveragePooling1D(shape[1]))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(15))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model
