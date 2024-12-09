import numpy as np

class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By default it's 1.0."""

    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias).""" 
        self.weights = (np.random.rand(inputs+1) * 2) - 1 
        self.bias = bias

    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        x_sum = np.dot(np.append(x,self.bias),self.weights)
        return self.sigmoid(x_sum)

    def set_weights(self, w_init):
        """Set the weights. w_init is a python list with the weights."""
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        """Evaluate the sigmoid function for the floating point input x."""
        return 1/(1+np.exp(-x))



class MultiLayerPerceptron:     
    """A multilayer perceptron class that uses the Perceptron class above.
       Attributes:
          layers:  A python list with the number of elements per layer.
          bias:    The bias term. The same bias is used for all neurons.
          eta:     The learning rate."""

    def __init__(self, layers, bias = 1.0):
        """Return a new MLP object with the specified parameters.""" 
        self.layers = np.array(layers,dtype=object)
        self.bias = bias
        self.network = [] # The list of lists of neurons
        self.values = []  # The list of lists of output values        
        
        for i in range(len(self.layers)):
            self.network.append([])
            self.values.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            if i>0:
                for j in range(self.layers[i]):
                    self.network[i].append(Perceptron(self.layers[i-1], bias))
            # self.network.append([Perceptron(layers[i], bias) for j in range(layers[i+1])])
            # self.values.append([0.0 for j in range(layers[i+1])])        
        self.network = np.array([np.array(x) for x in self.network],dtype=object)
        self.values = np.array([np.array(x) for x in self.values],dtype=object)
    def setWeights(self, w_init):
        """Set the weights. weights is a python list with the weights."""
        # weights for layer i , j nuron  = weights[i-1][j] of size k
        # so w_init will be a list of lists of lists   
        for i in range(1,len(self.network)):
            for j in range(len(self.network[i])):
                self.network[i][j].set_weights(w_init[i-1][j])
                self.network[i][j].bias = self.bias
    
    def printWeights(self):
        for i in range(1,len(self.network)):
            for j in range(len(self.network[i])):
                print("Layer",i+1,"Neuron",j,"Weights:",self.network[i][j].weights)

    def run(self, x):
        """Run the MLP. x is a python list with the input values."""
        x = np.array(x,dtype=object)
        self.values[0] = x
        # for layer i 1-->n: for neuron j 1-->m: run the neuron
        # weights of that neuron dot inputs of that layer + bias\
        Nlayers = len(self.network)
        for i in range(1,Nlayers):
            NNeurons = len(self.network[i])
            for j in range(NNeurons):
                currentNeuron = self.network[i][j]
                self.values[i][j]= currentNeuron.run(self.values[i-1])
        return self.values[-1]
    

#test code
mlp = MultiLayerPerceptron([2,2,1])
mlp.setWeights([[[-100,-100,150],[15,15,-10]],[[10,10,-15]]])
mlp.printWeights()
print(mlp.run([0,0]))
print(mlp.run([0,1]))
print(mlp.run([1,0]))
print(mlp.run([1,1]))
