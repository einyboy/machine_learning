#-*-coding:utf-8-*-
import numpy as np
import tensorflow as tf
import sklearn.preprocessing  as prep
from tensorflow.examples.tutorials.mnist import input_data

file_data = "MNIST_data/"
epochs = 10
display_step = 1
batch_szie = 128
mnist= input_data.read_data_sets(file_data, one_hot=True)
n_samples = int(mnist.train.num_examples)

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

def get_weigths(n_input, n_hidden):
		return tf.Variable(xavier_init(n_input,	n_hidden))
		
def get_baise(ndim):
	return tf.Variable(tf.zeros([ndim], dtype=tf.float32))

def standard_scale(X_train, X_test):
	preprocessor = prep.StandardScaler().fit(X_train)
	X_train = preprocessor.transform(X_train)
	X_test = preprocessor.transform(X_test)
	return X_train, X_test
	
def get_random_block_from_data(data, batch_szie):
	start_index = np.random.randint(0, len(data) - batch_szie)
	return data[start_index:(start_index + batch_szie)]

X_train, X_test = standard_scale(train_x, test_x)

class AdditivateGaussianNoiseAutoencoder(object):
	def __init__(self, n_input, n_hidden = [128], transfer_function=tf.nn.softplus,
				optimizer = tf.train.AdamOptimizer(), scale = 0.1):
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.transfer = transfer_function
		self.scale = tf.placeholder(tf.float32)
		self.training_scale = scale
		self.weights = self.initialize_weights()
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		
		self.hidden1 = self.transfer(
							tf.add(tf.matmul(
								self.x + scale * tf.random_normal((n_input,)),
								self.weights['w1']), self.weights['b1']))
								
		self.hidden = self.transfer(
							tf.add(tf.matmul(
								self.hidden1,
								self.weights['w2']), self.weights['b2']))						
								
		self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w3']),
							self.weights['b3'])
		
		self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
							self.reconstruction, self.x), 2.0
							))
		self.train_step = optimizer.minimize(self.cost)
		
		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)
	
	
		
	def initialize_weights(self):
		all_weights = dict()
		all_weights['w1'] = get_weigths(self.n_input, self.n_hidden[0])
		all_weights['b1'] = get_baise(self.n_hidden[0])
		all_weights['w2'] = get_weigths(self.n_hidden[0], self.n_hidden[1])
		all_weights['b2'] = get_baise(self.n_hidden[1])
		all_weights['w3'] = get_weigths(self.n_hidden[1], self.n_input)
		all_weights['b3'] = get_baise(self.n_input)
		
		return all_weights
	
	def partial_fit(self, X):
		cost, opt = self.sess.run([self.cost, self.train_step],
				feed_dict = {self.x:X, self.scale:self.training_scale})
				
		return cost
		
	def calc_total_cost(self, X):
		return self.sess.run(self.cost, 
				feed_dict = {self.x:X, 
				self.scale:self.training_scale})
				
	def transform(self, X):
		return self.sess.run(self.hidden, 
				feed_dict = {self.x:X, self.scale:self.training_scale})
				
	def generate(self, hidden = None):
		if hidden is None:
			hidden = np.random_normal(size = self.weights['b1'])
			
		return self.sess.run(self.reconstruction, 
				feed_dict = {self.hidden:hidden})
	
	def reconstruct(self, X):
		self.sess.run(self.reconstruction, 
				feed_dict = {self.x:X, self.scale:self.training_scale})
				
	def getWeights(self):
		return self.sess.run(self.weights['w1'])
		
	def getBiase(self):
		return self.sess.run(self.weights['b1'])
		
autoencoder = AdditivateGaussianNoiseAutoencoder(n_input = 784,
		n_hidden = [200,1000],
		transfer_function = tf.nn.softplus,
		#transfer_function = tf.nn.relu,
		#optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
		optimizer = tf.train.AdamOptimizer(),
		scale = 0.01)
		
for epoch in range(epochs):
	avg_cost = 0.
	total_batch = int(n_samples / batch_szie)
	for i in range(total_batch):
		batch_xs = get_random_block_from_data(X_train, batch_szie)
		
		cost = autoencoder.partial_fit(batch_xs)
		avg_cost += cost / n_samples * batch_szie
	
	if epoch % display_step == 0:
		print("Epoch:", '%04d' % (epoch + 1), "avg cost=", "{:.9f}".format(avg_cost))
	

total_cost = autoencoder.calc_total_cost(X_test)
print("Total cost:{}".format(total_cost))
	