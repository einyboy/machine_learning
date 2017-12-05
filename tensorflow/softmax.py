#-*-coding:cp936-*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


file_data = "MNIST_data/"
mnist= input_data.read_data_sets(file_data, one_hot=True)

train_x = mnist.train.images
train_y = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels
val_x = mnist.validation.images
val_y = mnist.validation.labels

print ("Shape train_x:{} train_y:{}".format(train_x.shape, train_y.shape))
print ("Shape test_x:{} test_y:{}".format(test_x.shape, test_y.shape))
print ("Shape val_x:{} val_y:{}".format(val_x.shape, val_x.shape))


