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
W_1 = tf.Variable(tf.zeros([784,512]))
b_1 = tf.Variable(tf.zeros([512]))

W_2 = tf.Variable(tf.zeros([512,256]))
b_2 = tf.Variable(tf.zeros([256]))

W_3 = tf.Variable(tf.zeros([256,128]))
b_3 = tf.Variable(tf.zeros([128]))

W_4 = tf.Variable(tf.zeros([128,64]))
b_4 = tf.Variable(tf.zeros([64]))

W_5 = tf.Variable(tf.zeros([64,128]))
b_5 = tf.Variable(tf.zeros([128]))

W_6 = tf.Variable(tf.zeros([128,256]))
b_6 = tf.Variable(tf.zeros([256]))

W_7 = tf.Variable(tf.zeros([256,512]))
b_7 = tf.Variable(tf.zeros([512]))

W_8 = tf.Variable(tf.zeros([512,784]))
b_8 = tf.Variable(tf.zeros([784]))

y_1 = tf.nn.sigmoid(tf.matmul(x, W_1) + b_1)
y_2 = tf.nn.sigmoid(tf.matmul(y_1, W_2) + b_2)
y_3 = tf.nn.sigmoid(tf.matmul(y_2, W_3) + b_3)
y_4 = tf.nn.sigmoid(tf.matmul(y_3, W_4) + b_4)
y_5 = tf.nn.sigmoid(tf.matmul(y_4, W_5) + b_5)
y_6 = tf.nn.sigmoid(tf.matmul(y_5, W_6) + b_6)
y_7 = tf.nn.sigmoid(tf.matmul(y_6, W_7) + b_7)
y = tf.nn.sigmoid(tf.matmul(y_7, W_8) + b_8)

#Get the function to minimize
loss = tf.nn.l2_loss(y - x)

#create a training step for (stochastic) gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

l2diff = tf.sqrt( tf.reduce_sum(tf.square(tf.subtract(x, y)), reduction_indices=1))
loss_mean = tf.reduce_mean(l2diff)

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