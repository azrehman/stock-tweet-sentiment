import numpy as np
import tensorflow as tf
from time import localtime, strftime
from preprocess import max_seq_length, X_train, X_test, y_train, y_test
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import TensorBoard

from sklearn.utils import class_weight

# y_ints = [y.argmax() for y in y_train]
# class_weights = class_weight.compute_class_weight(
#     'balanced', np.unique(y_ints), y_ints)
# class_weight_dict = dict(enumerate(class_weights))
# print(class_weights)

curr_time = strftime('time-%H-%M', localtime())
architecture = 'E(64)x32xD0.5x2'
MODEL_NAME = f'Stock-RNN-{architecture}-{curr_time}'
tensorboard = TensorBoard(log_dir=f'logs/{MODEL_NAME}')

model = Sequential()

num_words = len(np.unique(np.concatenate([X_train, X_test]))) + 50
model.add(Embedding(num_words, 64, input_length=max_seq_length))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, ))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(learning_rate=1e-4)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, batch_size=64, epochs=25,
          validation_data=(X_test, y_test),  callbacks=[tensorboard])

model.save(f'{MODEL_NAME}.model')
