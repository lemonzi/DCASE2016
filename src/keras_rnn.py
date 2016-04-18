"""
Recursive neural net using fancy LSTM and Keras
"""

from sklearn import base
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM


class RNN(base.BaseEstimator, base.ClassifierMixin):

  def __init__(self, foo='bar'):
    pass

  def __getstate__(self):
    self.model_.save_weights(self.filename, overwrite=True)
    state = dict(self.__dict__)
    del state['model_']
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self.model_ = self._create_model()
    self.model_.load_weights(self.filename)

  def fit(self, X, y):
    self.model_ = self._create_model()
    self.model_.fit(X, y.toarray(), nb_epoch=50, verbose=1, show_accuracy=True)

  def predict(self, X):
    return self.model_.predict_classes(X)[0]

  def set_filename(self, filename):
    self.filename = filename

  def _create_model(self):
    model = Sequential()
    model.add(LSTM(1024, return_sequences=True, input_dim=60))
    model.add(Dropout(0.2))
    model.add(LSTM(1024, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(15))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model
