import time
import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep

# THIS PERFORMS FORECASTS OF DENGUE CASES IN NATIONAL CAPITAL REGION,
# PHILIPPINES USING RNNs AND TENSORFLOW


def sklearn_sscaler(xtrain, xtest):
    """
    This performs sklearn's StandardScaler [1] on training and test sets

    [1] https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    """
    train_samples, train_nx, train_ny = xtrain.shape
    test_samples, test_nx, test_ny = xtest.shape
    xtrain = xtrain.reshape((train_samples, train_nx * train_ny))
    xtest = xtest.reshape((test_samples, test_nx * test_ny))
    preprocessor = prep.StandardScaler().fit(xtrain)
    xtrain = preprocessor.transform(xtrain)
    xtest = preprocessor.transform(xtest)
    xtrain = xtrain.reshape((train_samples, train_nx, train_ny))
    xtest = xtest.reshape((test_samples, test_nx, test_ny))
    
    return xtrain, xtest

def preprocessor(dengue, n_seq, frac_train=0.75):
    """
    Preprocesses the dataset before plugging the former into the neural network.
    """
    number_features = len(dengue.columns)
    data = dengue.values
    
    sequence_length = n_seq + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index : index + sequence_length])
        
    result = np.array(result)
    row = round(frac_train * result.shape[0])
    train = result[: int(row), :]
    
    train, result = sklearn_sscaler(train, result)    
    
    TRAIN_X = train[:, : -1]
    TRAIN_Y = train[:, -1][: ,-1]
    TEST_X = result[int(row) :, : -1]
    TEST_Y = result[int(row) :, -1][ : ,-1]

    TRAIN_X = np.reshape(TRAIN_X, (TRAIN_X.shape[0], TRAIN_X.shape[1], number_features))
    TEST_X = np.reshape(TEST_X, (TEST_X.shape[0], TEST_X.shape[1], number_features))  

    return [TRAIN_X, TRAIN_Y, TEST_X, TEST_Y]   


def build_model(layers, dropout_rate=0.5, activation='linear',loss='mse',optimizer='rmsprop',metrics=['accuracy']):
    """
    A simple RNN will be built with 2 LSTM layers:

        LSTM -> Dropout -> LSTM --> Dropout --> Dense

    Activation from: https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/keras/activations.py

    Default values:
    Dropout rate: 0.5
    Activation function for hidden and output layers: Linear
    Loss function: MSE
    Optimizer: RMSProp
    Metrics: Accuracy
    """
    
    model = Sequential()

    model.add(LSTM(input_dim=layers[0], output_dim=layers[1],return_sequences=True))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(dropout_rate))

    model.add(Dense(output_dim=layers[3]))
    model.add(Activation(activation))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model    


####################
# intialization
fname='D-NCR.csv' # this loads weekly dengue cases for NCR from Jan 2008 to Mar 2016
df=pd.read_csv(fname)
df=df[['D']] # get the dengue cases only


window=20; activation='linear'
TRAIN_X, TRAIN_Y, TEST_X, TEST_Y = preprocessor(df[:: -1], window)
model = build_model([TRAIN_X.shape[2], window, 100, 1], activation=activation)

# training the model
batch_size=10; epochs=5; validation_split=0.2; verbose=0
model.fit(TRAIN_X, TRAIN_Y,batch_size,epochs,validation_split,verbose)


TRAIN_SCORE = model.evaluate(TRAIN_X, TRAIN_Y, verbose=0)
print('Score for training set: MSE=%.2f, RMSE=%.2f' % (TRAIN_SCORE[0], math.sqrt(TRAIN_SCORE[0])))

TEST_SCORE = model.evaluate(TEST_X, TEST_Y, verbose=0)
print('Score for test set: MSE=%.2f, RMSE=%.2f' % (TEST_SCORE[0], math.sqrt(TEST_SCORE[0])))

# generating predictions
diff,ratio = [],[]
PREDS = model.predict(TEST_X)
for u in range(len(TEST_Y)):
    pr = PREDS[u][0]
    ratio.append((TEST_Y[u] / pr) - 1)
    diff.append(abs(TEST_Y[u] - pr))

# plotting
import matplotlib
import matplotlib.pyplot as plt

plt.plot(PREDS, color='black', label='Predicted')
plt.plot(TEST_Y, color='green', label='Actual')
plt.legend(loc='upper left')
plt.savefig('D-NCR-results.png', dpi=300, bbox_inches='tight')
