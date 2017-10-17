import numpy as np
from scipy.special import expit

#the loss function ||y - \overline{y}||^{2}
#yy -> an array of predictions, where the 0th position gives the prediction
def loss(yy, y):
	ans = (y - yy[0])**2
	return ans

def dloss(x,y):
	return 2*(x-y)

class autoEncoder:

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
		i = len(self.layers) - 1
		returned, xs, vals = self.forward(x)
		if layer[i][3] == expit:
			delta = dloss(returned, y) * \ dsigma(vals[-1])
			self.diffs[i][0] += np.dot(delta, xs[-2].T)
		for layer in self.layers[::-1]:
			if i == len(self.layers) - 1:
				pass
			if layer[3] == expit:
				val = vals[i]
				ds = dsigma(val)
				delta = np.dot(layer[0].T, delta) * ds
				self.diffs[i][0] += np.dot(delta, xs[i].T)
				self.diffs[i][1] += delta
			i -= 1
	
	def __update_weights(self, mu = 1):
		i = 0
		for layer in self.layers:
			layer[0] -= mu * self.diffs[i][0]
			layer[1] -= mu * self.diffs[i][1]
			self.diffs[i][0] = np.zeros_like(layer[0])
			self.diffs[i][1] = np.zeros_like(layer[1])
			i += 1
	def update(self, x, y=None):
		if y is None:
			y = x
		self.backward(x,y)
		self.__update_weights()
		return loss
		
		
encoder = autoEncoder(3)
x = np.array([1,2,3])
encoder.addLayer()
encoder.addLayer(nneurons=10)
encoder.addLayer(nneurons=3)