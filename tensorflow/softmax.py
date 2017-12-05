#-*-coding:cp936-*-

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import sklearn.preprocessing as prep

file_data = "MNIST_data/"
epochs = 1000
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

def xavier_init(fan_in, fan_out, constant = 1):
	low = -constant * np.sqrt(6.0/ (fan_in + fan_out))
	high = constant * np.sqrt(6.0/ (fan_in + fan_out))
	return tf.random_uniform((fan_in, fan_out), 
								minval = low, maxval = high,
								dtype = tf.float32)

def standard_scale(X_train, X_test):
	preprocessor = prep.StandardScaler().fit(X_train)
	X_train = preprocessor.transform(X_train)
	X_test = preprocessor.transform(X_test)
	return X_train, X_test
	
sess = tf.InteractiveSession()
X = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
#W = tf.Variable(tf.zeros([784, 10]))
W = tf.Variable(xavier_init(784, 10))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(X, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_predict = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

tf.global_variables_initializer().run()

for i in range(epochs):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	val_xs, val_ys = mnist.validation.next_batch(100)
	batch_xs, val_xs = standard_scale(batch_xs, val_xs)
	_, acc = sess.run([train_step,accuracy], feed_dict = {X: batch_xs, y_: batch_ys})
	val_acc = accuracy.eval({X: val_xs, y_: val_ys})
	if i % 100 == 0:
		print ("{}/{} train:{} val:{}".format(i, epochs, acc, val_acc))

test_x, _ = standard_scale(test_x, val_xs)
test_accuracy = accuracy.eval({X: test_x, y_: test_y})

print ("=====================================")
print ("test acc:{}".format(test_accuracy))

sess.close()




