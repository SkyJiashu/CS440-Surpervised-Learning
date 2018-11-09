# mlp.py
# -------------

# mlp implementation
import numpy as np
import random

PRINT = True

class MLPClassifier:
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mlp"
        self.max_iterations = max_iterations
        self.sizes = [784, 100, 10]
        self.num_layers = len(self.sizes)
        
        self.weights_hidden = []
        self.weights_output = []
        for i in range(0, self.sizes[1]):
            t = [random.uniform(-0.05, 0.05) for i in range(0, self.sizes[0])]
            self.weights_hidden.append(t)
        for i in range(0, self.sizes[0]):
            t = [random.uniform(-0.05, 0.05) for i in range(0, self.sizes[1])]
            self.weights_output.append(t)
            
    def transferToList(self, trainingData):
        b = []
        for i in range(len(trainingData)):
            b.append([])
            for x in range(28):
                for y in range(28):
                    b[i].append(trainingData[i][(x, y)])
        
        return b

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        sum = 0.0
        aj = []
        ai = []
        for j in range(0, self.sizes[1]):
            for k in range(0, self.sizes[0]):
                sum += self.weights_hidden[j][k] * a[k]
            aj.append(sigmoid(sum))
            sum = 0.0
        
        for i in range(0, self.sizes[2]):
            for j in range(0, self.sizes[1]): 
                sum += self.weights_output[i][j] * aj[j]
            ai.append(sigmoid(sum))
            sum = 0.0
        return ai

    def train( self, trainingData, trainingLabels, validationData, validationLabels):
        learningRate = 0.08
        trainingData = self.transferToList(trainingData)
        data = [[None for _ in range(2)]for _ in range(len(trainingData))]
        for i in range(0, len(trainingData)):
            data[i][0] = trainingData[i]
            data[i][1] = trainingLabels[i]
        for iteration in range(self.max_iterations):
            random.shuffle(data)
            print "Starting iteration ", iteration, "..."
            for x, y in data:  
                self.backprop(x, y, learningRate) 

    def backprop(self, x, y, learningRate):
        #Feedforward
        ai = self.feedforward(x)
        
        #Backward
        error = []
        value = 0.0
        for i in range(0, self.sizes[2]):
            if i == y:
                value=1.0
            else:
                value=0.0
            error.append(value - ai[i])
        delta_i = []
        for i in range(0, self.sizes[2]):
            delta_i.append(error[i] * sigmoid_prime(ai[i]))
        aj = []
        sum = 0.0
        for j in range(0, self.sizes[1]):
            for k in range(0, self.sizes[0]):
                sum += self.weights_hidden[j][k] * x[k]
            aj.append(sigmoid(sum))
            sum = 0.0
        
        #Updating the weights in the output layer
        for j in range(0, self.sizes[1]):
            for i in range(0, self.sizes[2]):
                self.weights_output[i][j] += learningRate * aj[j] * delta_i[i]
        
        delta_j = []
        for j in range(0, self.sizes[1]):
            for i in range(0, self.sizes[2]):
                sum += self.weights_output[i][j] * delta_i[i]
            delta_j.append(sigmoid_prime(aj[j]) * sum)
            sum = 0.0
                
        #Updating the weights in the hidden layer
        for k in range(0, self.sizes[0]):
            for j in range(0, self.sizes[1]):
                self.weights_hidden[j][k] += learningRate * x[k] * delta_j[j]
        
    def classify(self, data):
        data = self.transferToList(data)
        guesses = []
        for l in data:
            guesses.append(np.argmax(self.feedforward(l)))
        return guesses

#### Miscellaneous functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))