import numpy as np
from scipy.special import expit

#the loss function ||y - \overline{y}||^{2}
#yy -> an array of predictions, where the 0th position gives the prediction
def loss(yy, y):
	ans = (y - yy[0])**2
	return ans

#the derivative of the loss function
def dloss(x,y):
	return 2*(x-y)

#the derivative of the sigmoid function
def dsigma(x):
	ans = expit(x)**2 * np.exp(-x)
	return ans

#a paramter for turning on denoising
denoise = True

def corrupt(x):
	'''Introduce a uniformly distributed random decay to our input
	
	params:
		x -> the value t be corrupted
	'''
	global denoise
	if denoise:
		ret = x * np.random.uniform(low = -1,high = 1,size = x.shape)
	else:
		ret = x
	return ret

class autoEncoder:
	'''A naive implementation of an autoencoder'''

	def __init__(self, dim):
		self.layers = []
		self.diffs = []
		self.inDim = dim
		
	#creates a triple (for a layer) containing:
	#0 -> The weight matrix
	#1 -> The bias
	#2 -> the activation function
	#and a double (for weight diffs) containing:
	#0 -> Weight matrix diff
	#1 -> Bias diff
	def addLayer(self, nneurons = 5, activation = "SIGMOID"):
		'''Adds a layer to the Autoencoder
		
		params:
			nneurons -> the number of neurons to the layer (defult to 5)
			activation -> A key for the activation function for the layer, defult SIGMOID for the sigmoid function
		'''
		triple = []
		if len(self.layers) == 0:
			triple.append(np.random.randn(nneurons, self.inDim))
		else:
			triple.append(np.random.randn(nneurons, self.layers[len(self.layers) - 1][0].shape[0]))
		triple.append(np.zeros(nneurons))
		if activation == "SIGMOID":
			triple.append(expit)
		self.layers.append(triple)
		self.diffs.append([np.zeros_like(self.layers[len(self.layers)-1][0]), np.zeros_like(self.layers[len(self.layers)-1][1])])
	
	def forward(self, x):
		'''Performs a forward pass of the network
		
		params:
			x -> The value to be fed through the network
		returns:
			x -> the value pushed out of the network
			xs -> the output of each layer (i.e. sigma(Wx + b))
			vals -> the value of the matrix pass at each layer (ie Wx+b)
		'''
		if len(self.layers) == 0:
			raise ValueError("the network needs layers")
		if self.layers[len(self.layers) - 1][0].shape[0] != len(x):
			raise ValueError("The network must be built to output something of the same length")
		xs = []
		vals = []
		for layer in self.layers:
			val = np.dot(layer[0],x)
			val += layer[1]
			vals.append(val)
			x = layer[2](val)
			xs.append(x)
		return x,xs, vals
	
	def backward(self, x,y):
		'''Performs a backward pass of the network, and caluclates the diffs.
		
		params:
			x -> the value to be fed in to the network
			y -> The value that should be returned by the network
		'''
		i = len(self.layers) - 1
		returned, xs, vals = self.forward(x)
		if self.layers[i][2] == expit:
			delta = dloss(returned, y) * dsigma(vals[-1])
			self.diffs[i][0] += np.dot(delta, xs[-1].T)
			self.diffs[i][1] += delta
		for layer in self.layers[::-1]:
			if i == len(self.layers) - 1:
				i -= 1
				pass
			if layer[2] == expit:
				val = vals[i]
				ds = dsigma(val)
				delta = np.dot(layer[0].T, delta) * ds
				self.diffs[i][0] += np.dot(delta, xs[i].T)
				self.diffs[i][1] += delta
			i -= 1
	
	def __update_weights(self, mu = 1):
		'''Updates the weights/biases of the network, and resets the diffs
		
		params:
			mu -> the learning rate
		'''
		i = 0
		for layer in self.layers:
			layer[0] -= mu * self.diffs[i][0]
			layer[1] -= mu * self.diffs[i][1]
			self.diffs[i][0] = np.zeros_like(layer[0])
			self.diffs[i][1] = np.zeros_like(layer[1])
			i += 1
	
	def update(self, x, y=None):
		'''Update the network
		
		params:
			x -> the value to be fed in to the network
			y -> the value expected (if not specified, will assume it's equal to x)
		'''
		if y is None:
			y = x
		self.backward(x,y)
		self.__update_weights()
		return loss
		
def main():		
	encoder = autoEncoder(3)
	x = np.array([1,2,3])
	xx = corrupt(x)
	encoder.addLayer()
	encoder.addLayer(nneurons=10)
	encoder.addLayer(nneurons=3)
	for _ in range(100000):
		encoder.update(x, y=xx)
	
if __name__ == '__main__':
	main()