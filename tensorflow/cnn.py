#-*-coding:utf-8-*-

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

file_data = "MNIST_data/"
epochs = 3000
display_tep = 20
batch_size = 64
drop_prob = 0.5
mnist= input_data.read_data_sets(file_data, one_hot=True)
test_x = mnist.test.images
test_y = mnist.test.labels

sess = tf.InteractiveSession()

def weigth_varibale(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
	
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1],
				padding='SAME')
				
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],
				padding='SAME')
				
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])
W_conv1 = weigth_varibale([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weigth_varibale([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weigth_varibale([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weigth_varibale([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv), 
			reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_predict = tf.equal(tf.argmax(y_, 1),tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

tf.global_variables_initializer().run()

for epoch in range(epochs):
	batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	train_step.run({x:batch_xs, y_:batch_ys, keep_prob:drop_prob})
	
	if epoch % display_tep == 0:
		acc_rate = accuracy.eval({x: batch_xs, y_: batch_ys, keep_prob:1.0})
		print ("======{}/{} acc:{}".format(epoch, epochs, acc_rate))

test_accuracy = accuracy.eval({x:test_x[:1000], y_:test_y[:1000], keep_prob:1.0})

print ("=====================================")
print ("test acc:{}".format(test_accuracy))
sess.close()