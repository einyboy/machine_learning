import os
import random

# suppress the TensorFlow warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as  np
from keras.models import load_model


def encode(X,seq_len, vocab_size):
    x = np.zeros((len(X),seq_len, vocab_size), dtype=np.float32)
    for ind,batch in enumerate(X):
        for j, elem in enumerate(batch):
            x[ind, j, elem] = 1
    return x

model = load_model('model.h5')

seq_len = 15 # number of elements in sequence to sort
max_no = 100 # upper range of the numbers in sequence
verbose = False
win_cnt = 0
numb_trials = 1000

for ii in range(numb_trials):
    testX = np.random.randint(max_no, size=(1, seq_len))
    test = encode(testX, seq_len, max_no)
    y = model.predict(test, batch_size=1)
    #print(testX)
    np_sorted = np.sort(testX)[0]
    rnn_sorted = np.argmax(y, axis=2)[0]
    is_equal = np.array_equal(np_sorted, rnn_sorted)
    if is_equal: win_cnt += 1
    if verbose:
        print(np_sorted, ': sorted by NumPy algorithm')
        print(rnn_sorted, ': sorted by trained RNN')
        print("\n")

print('\nSuccess Rate: {0:.2f} %'.format(100*win_cnt/numb_trials))