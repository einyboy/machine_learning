#-*-coding:utf-8-*-
import numpy as np
import scipy as sp
from keras.models import Sequential
from keras.layers import Input, Dense
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

NUM_DIGITS = 10
def binary_encode(i, num_digits):
	bin = [i >> d & 1 for d in range(num_digits)]
	bin.reverse()
	#print(bin)
	return np.array(bin)

def odd_even_encode(i):
    if i % 2 == 0:
        return np.array([0, 1])
    else:
        return np.array([1, 0])
        
def odd_even_decode(num,predict):
    return {num:['odd','even'][predict]}
    
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(301, 400)])
trY = np.array([odd_even_encode(i) for i in range(301, 400)])
testX = np.array([binary_encode(i, NUM_DIGITS) for i in range(1000,1300)])
testY = np.arange(1000,1300)

def keras_build():
    model = Sequential()
    model.add(Dense(2, input_dim = NUM_DIGITS , activation = 'softmax'))
    return model

def kera_model():
    model = keras_build()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit(trX, trY, nb_epoch=300, batch_size=10)
    predictY = model.predict(testX)
    lable = np.argmax(predictY, 1)
    lable = np.vectorize(odd_even_decode)(testY,lable)
    print(lable)
    print(model.summary())
    
def sklearn_model():
    clf = LogisticRegression(penalty='l2', C=1000.0, random_state=0)
    label = np.argmax(trY,1)
    clf.fit(trX, label)
    lable = clf.predict(testX)
    lable = np.vectorize(odd_even_decode)(testY,lable)
    print(lable)
    print(clf.coef_)
    print(clf.intercept_)
    
if __name__=='__main__':
    #kera_model()
    sklearn_model()
    