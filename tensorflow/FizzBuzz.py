#-*-coding:utf-8-*-

import numpy as np
import tensorflow as tf

epochs = 8000
display_tep = 20
batch_size = 128

def binary_encode(i, num_digits):
	bin = [i >> d & 1 for d in range(num_digits)]
	bin.reverse()
	#print(bin)
	return np.array(bin)
	
def fizz_buzz_encode(i):
	if i % 15 == 0: return np.array([0, 0, 0, 1]) # FizzBuzz
	elif i % 5 == 0: return np.array([0, 0, 1, 0]) # Buzz
	elif i % 3 == 0: return np.array([0, 1, 0, 0]) # Fizz
	else: return np.array([1, 0, 0, 0])

def init_weigths(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))
	
def init_biases(shape):
	return tf.Variable(tf.zeros(shape))

def get_batch_block(X, Y, batch_size):
	start_index = np.random.randint(0, len(X) - batch_size)
	batch_xs = X[start_index:(start_index + batch_size)]
	batch_ys = Y[start_index:(start_index + batch_size)]
	return batch_xs, batch_ys

def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

def model():
	
	W1 = init_weigths([NUM_DIGITS, NUM_HIDDEN])
	b1 = init_biases([NUM_HIDDEN])
	W2 = init_weigths([NUM_HIDDEN, 4])
	b2 = init_biases([4])
	hidden = tf.nn.relu(tf.matmul(X, W1) + b1)
	py_x = tf.nn.softmax(tf.matmul(hidden, W2) + b2)
	return py_x
	
NUM_DIGITS = 12
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = np.array([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

#train_xs = np.array([binary_encode(i, NUM_DIGITS) for i in range(1, 6)])
#train_ys = np.array([fizz_buzz_encode(i) for i in range(1, 6)])

NUM_HIDDEN = 100
sess = tf.InteractiveSession()
X = tf.placeholder(tf.float32, [None, NUM_DIGITS])
Y = tf.placeholder(tf.float32, [None, 4])

'''
w_h = init_weigths([NUM_DIGITS, NUM_HIDDEN])
w_o = init_weigths([NUM_HIDDEN, 4])
py_x = model(X, w_h, w_o)
'''
py_x = model()

cross_entropy = tf.reduce_mean(-tf.reduce_sum(
				Y * tf.log(py_x),reduction_indices = [1]))
'''
cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=py_x))
'''			
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

cross_predict = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1))
accuracy = tf.reduce_mean(tf.cast(cross_predict, tf.float32))

predict = tf.argmax(py_x, 1)

tf.global_variables_initializer().run()
for epoch in range(epochs):
	#Extensive training sample by permutation
	p = np.random.permutation(range(len(trX)))
	trX, trY = trX[p], trY[p]
	
	for start in range(0, len(trX), batch_size):
		end = start + batch_size
		_, acc_rate = sess.run([train_step,accuracy], 
					feed_dict = {X:trX[start:end], Y:trY[start:end]})
	'''
	
	batch_xs, batch_ys = get_batch_block(trX, trY, batch_size)
	_, acc_rate = sess.run([train_step, accuracy], feed_dict = {X:batch_xs, Y:batch_ys})
	'''
	
	if epoch % display_tep == 0:
		print ("======{}/{} acc:{}".format(epoch, 
				epochs, acc_rate))


numbers = np.arange(1023, 2000)
teX = np.transpose(binary_encode(numbers, NUM_DIGITS))
teY = sess.run(predict, feed_dict={X: teX})
output = np.vectorize(fizz_buzz)(numbers, teY)
print(output)


sess.close()