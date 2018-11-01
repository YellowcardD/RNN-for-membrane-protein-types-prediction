import numpy as np
from utils import *
from keras.utils.np_utils import to_categorical
import pandas as pd
from keras.layers import Dense, LSTM, Activation, Input, Dropout, BatchNormalization, Conv1D
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.optimizers import Adam, SGD
from keras.utils import normalize
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from keras.layers import Bidirectional
from keras.layers import Masking
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU


decay = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, mode='max')
batch_sizes = [32, 64] # number of training examples are fed to one iteration
LSTM_units = [128, 256] # number of units in LSTM
filters = [128, 196, 256] # number of filters in Conv1D layer


def model_selection():
    """
    We search for optimal hyperparameters for model using grid search.
    """
    i = 0
    for batch_size in batch_sizes:
        for units in LSTM_units:
            for filter in filters:
                i = i + 1
                model = create_model(input_shape=[1000, 20], units=units, filters = filter)
                adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0)
                model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
                checkpoint = ModelCheckpoint('model/model_weights_v9_%d.h5' %i, monitor='val_acc', save_best_only=True,
                                             save_weights_only=True, mode='max')
                hist = model.fit(X, Y_oh, epochs=25, batch_size=batch_size, validation_data=(X_val, Y_val_oh),
                                 callbacks=[checkpoint, decay])


def create_model(input_shape, units, filters):
    """
    Create our final model instance.
    :param input_shape: the shape of input tensor
    :param units: number of units in LSTM
    :param filters: number of filters in Conv1D layer
    :return: a Keras model instance
    """
    X_input = Input(shape=input_shape, dtype='float32')

    X = Conv1D(filters=filters, strides=10, kernel_size=15, padding='same')(X_input)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Masking(mask_value=0.0)(X)
    X = Bidirectional(LSTM(units=units, return_sequences=True))(X)
    X = BatchNormalization()(X)
    X = Dropout(0.5)(X)

    X = Bidirectional(LSTM(units=units, return_sequences=False))(X)
    X = Dropout(0.5)(X)
    X = BatchNormalization()(X)

    X = Dropout(0.5)(X)
    X = Dense(8)(X)
    X = Activation('softmax')(X)

    model = Model(X_input, X)

    return model

# We load our dataset from files, the procedure of saving file was done by utils.py
X = np.load('X.npy')
X = X[:, :1000, :]
Y = np.load('Y.npy')

# Split the dataset to training set and validation set
X, X_val, Y, Y_val = train_test_split(X, Y, test_size=0.2, random_state=33, stratify=Y)
# Transform Y to one-hot encoding form for subsequent training and evaluating
Y_oh = to_categorical(Y, 8)
Y_val_oh = to_categorical(Y_val, 8)

#model_selection()

# read test set from files
Y_test = np.load('Y_test.npy')
Y_test_oh = to_categorical(Y_test)
X_test = np.load('X_test.npy')
X_test = X_test[:, :1000, :]

model = create_model(input_shape=[1000, 20], units=128, filters=256)
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# load our trained weight to model
model.load_weights('E:\PycharmProjects\RNN-for-protein\model\model_weights_v9_3.h5')

# Evaluate our model using test set
loss, acc = model.evaluate(X_test, Y_test_oh)
print(loss, acc) # 0.327003745763 0.940918532401