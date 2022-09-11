import pandas as pd
import numpy as np

import os
import sys

import librosa
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization ,Activation, LSTM
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

################################### First model
def make_model(X_train):
  # building the model:
  model = Sequential()
  model.add(Conv1D(256, 8, padding='same',activation = 'relu',input_shape=(X_train.shape[1],1)))  
  model.add(Conv1D(256, 8, padding='same', activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))
  model.add(MaxPooling1D(pool_size=(8)))
  model.add(Conv1D(128, 8, padding='same', activation='relu'))
  model.add(Conv1D(128, 8, padding='same', activation='relu'))
  model.add(Dropout(0.4))
  model.add(Conv1D(128, 8, padding='same', activation='relu'))
  model.add(Conv1D(128, 8, padding='same', activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))
  model.add(MaxPooling1D(pool_size=(8)))
  model.add(Conv1D(64, 8, padding='same', activation='relu'))
  model.add(Conv1D(64, 8, padding='same', activation='relu'))
  model.add(Flatten())

  model.add(Dense(7, activation='softmax')) 
  opt = tf.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer= opt ,loss='categorical_crossentropy',metrics=['acc'])
  return model


# loading the features  
Features = pd.read_csv('/features.csv', sep=',')
Features.head()

X = Features.iloc[: ,:-1].values
Y = Features['labels'].values

encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()


kf = KFold(n_splits=10, shuffle=True, random_state=0)
accuracies = []
i = 0
for train_index, test_index in kf.split(X):
    print("**************************************** Fold : {0} ****************************************".format(i))
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # making our data compatible to model.
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    
    model = make_model(X_train)
    checkpointer = ModelCheckpoint('/best models/CNN/10Fold/ser_cnn_'+str(i)+'.h5',monitor='val_acc', mode='max', verbose=2, save_best_only=True)
    #checkpointer = ModelCheckpoint('',monitor='val_acc', mode='max', verbose=2, save_best_only=True)
    model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test), callbacks=[checkpointer])
    loss, accuracy = model.evaluate(X_test,y_test)
    accuracies.append(accuracy)
    print("Loss: {0} | Accuracy: {1}".format(loss, accuracy))
    i+=1
print("Average 10 Folds Accuracy: {0}".format(np.mean(accuracies)))
print("STD: {0}".format(np.std(accuracies)*100))

################################### The second model

def make_modelLSTM(X_train):
  model_lstm = Sequential()
  model_lstm.add(LSTM(128, input_shape=(X_train.shape[1],1)))  # (time_steps = 1, n_feats)
  model_lstm.add(Dropout(0.5))
  model_lstm.add(Dense(32, activation='relu'))
  model_lstm.add(Dense(7, activation='softmax')) 
  optimzer = Adam(lr=0.001)
  model_lstm.compile(loss='categorical_crossentropy', optimizer=optimzer, metrics=['accuracy'])
  return model_lstm


X = Features.iloc[: ,:-1].values
Y = Features['labels'].values


encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()


kf = KFold(n_splits=10, shuffle=True, random_state=0)
accuracies = []
i = 0
for train_index, test_index in kf.split(X):
    print("**************************************** Fold : {0} ****************************************".format(i))
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # making our data compatible to model.
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    
    model = make_modelLSTM(X_train)
    path = '/best models/LSTM/10Fold/ser_lstm_'+str(i)+'.h5'
    checkpointer1 = ModelCheckpoint(path, monitor='val_acc', mode='max', verbose=2, save_best_only=True)
    model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test), callbacks=[checkpointer1])
    loss, accuracy = model.evaluate(X_test,y_test)
    accuracies.append(accuracy)
    print("Loss: {0} | Accuracy: {1}".format(loss, accuracy))
    i+=1
print("Average 10 Folds Accuracy: {0}".format(np.mean(accuracies)))
print("STD: {0}".format(np.std(accuracies)*100))


################################### The third model

def make_modelCNN_SVM(X_train):
  model_cnn_svm = Sequential()
  model_cnn_svm.add(Conv1D(256, 8, padding='same',activation = 'relu',input_shape=(X_train.shape[1],1)))  
  model_cnn_svm.add(Conv1D(256, 8, padding='same', activation='relu'))
  model_cnn_svm.add(BatchNormalization())
  model_cnn_svm.add(Dropout(0.4))
  model_cnn_svm.add(MaxPooling1D(pool_size=(8)))
  model_cnn_svm.add(Conv1D(128, 8, padding='same', activation='relu'))
  model_cnn_svm.add(Conv1D(128, 8, padding='same', activation='relu'))
  model_cnn_svm.add(Dropout(0.4))
  model_cnn_svm.add(Conv1D(128, 8, padding='same', activation='relu'))
  model_cnn_svm.add(Conv1D(128, 8, padding='same', activation='relu'))
  model_cnn_svm.add(BatchNormalization())
  model_cnn_svm.add(Dropout(0.4))
  model_cnn_svm.add(MaxPooling1D(pool_size=(8)))
  model_cnn_svm.add(Conv1D(64, 8, padding='same', activation='relu'))
  model_cnn_svm.add(Conv1D(64, 8, padding='same', activation='relu'))
  model_cnn_svm.add(Flatten())

  model_cnn_svm.add(Dense(7, kernel_regularizer=tf.keras.regularizers.l2(0.01),activation='softmax'))
  opt = tf.keras.optimizers.Adam(learning_rate=0.001)
  model_cnn_svm.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
  return model_cnn_svm


X = Features.iloc[: ,:-1].values
Y = Features['labels'].values


encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()


kf = KFold(n_splits=10, shuffle=True, random_state=0)
accuracies = []
i = 0
for train_index, test_index in kf.split(X):
    print("**************************************** Fold : {0} ****************************************".format(i))
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = make_modelCNN_SVM(X_train)
    path_cnn_svm = '/best models/CNN_SVM/10Fold/ser_cnn_svm_'+str(i)+'.h5'
    checkpointer = ModelCheckpoint(path_cnn_svm, monitor='val_acc', mode='max', verbose=2, save_best_only=True)
    model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test), callbacks=[checkpointer])
    loss, accuracy = model.evaluate(X_test,y_test)
    accuracies.append(accuracy)
    print("Loss: {0} | Accuracy: {1}".format(loss, accuracy))
    i+=1
print("Average 10 Folds Accuracy: {0}".format(np.mean(accuracies)))
print("STD: {0}".format(np.std(accuracies)*100))