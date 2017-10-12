from tensorflow.examples.tutorials.mnist import input_data

#not sure this is working the way I think it is
#if it is it's a really poor model

#don't think I need one_hot, since I'm training an autoencoder
#not attempting to learn the classification
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

import tensorflow as tf

#a placeholder is used to store values that won't change
#we'll load this with the MNIST data
#None indicates we'll take arbitrarily many entries in that dimension, na d784 is the size of an individual MNIST image
x = tf.placeholder(tf.float32, [None, 784])

#build our neural network
W_1 = tf.Variable(tf.zeros([784,400]))
b_1 = tf.Variable(tf.zeros([400]))

W_2 = tf.Variable(tf.zeros([400,100]))
b_2 = tf.Variable(tf.zeros([100]))

W_3 = tf.Variable(tf.zeros([100,784]))
b_3 = tf.Variable(tf.zeros([784]))

y_1 = tf.nn.sigmoid(tf.matmul(x, W_1) + b_1)
y_2 = tf.nn.sigmoid(tf.matmul(y_1, W_2) + b_2)
y = tf.nn.sigmoid(tf.matmul(y_2, W_3) + b_3)

#Get the function to minimize
loss = tf.nn.l2_loss(y - x)

#create a training step for (stochastic) gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

l2diff = tf.sqrt( tf.reduce_sum(tf.square(tf.subtract(x, y)), reduction_indices=1))
loss_mean = tf.reduce_mean(loss)

with tf.Session() as sess:
	#initialize our Variables
	sess.run(tf.global_variables_initializer())
	for _ in range(10000):
		#load the next 50 values from MNIST (hre is where the stochastic comes in)
		batch_x, batch_y = mnist.train.next_batch(50)
		#run our training step (each Placeholder needs a value in the dictionary)
		train_step.run(feed_dict={x: batch_x})
	#get the accuracy
	print(sess.run(loss_mean, feed_dict={x: mnist.test.images}))