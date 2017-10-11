import numpy as np
from scipy.special import expit

#the loss function ||y - \overline{y}||^{2}
#yy -> an array of predictions, where the 0th position gives the prediction
def loss(yy, y):
	ans = (y - yy[0])**2
	return ans

class autoEncoder:

	def __init__(self, dim):
		self. layers = []
		self.inDim = dim
		
	#creates a triple containing:
	#0 -> The weight matrix
	#1 -> The bias
	#2 -> the activation function
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
	
	def forward(self, x):
		if len(self.layers) == 0:
			raise ValueError("the network needs layers")
		if self.layers[len(self.layers) - 1][0].shape[0] != len(x):
			raise ValueError("The netrok must be built to output something of the same length")
		for layer in self.layers:
			print(layer[0])
			print(x)
			val = np.dot(layer[0],x)
			val += layer[1]
			#TODO val logic?
			x = layer[2](val)
		return x
		
encoder = autoEncoder(3)
x = np.array([1,2,3])
encoder.addLayer()
encoder.addLayer(nneurons=10)
encoder.addLayer(nneurons=3)
out = encoder.forward(x)
print(out)