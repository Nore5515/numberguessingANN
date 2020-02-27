# The network will take in a letter convereted to binary.
# It will try to guess the letter. If it guesses correctly, then yay!
# Otherwise, boo.

# INPUT - 8 NEURONS OF 1 OR 0
# TWO HIDDEN LAYERS OF 8 EACH
# OUTPUT - 26 NEURONS, ONE FOR EACH LETTER

import math
import random
import string

def Sigmoid (input):
    output = 1/(1 + math.exp(-input)) 
    return output

class Neuron:
    activation = 0.0
    weights = []
    
    def __init__ (self):
        pass
    
    def SetWeights(self, inWeights):
        self.weights = inWeights
        
    def RandomWeights(self, count):
        self.weights = []
        for x in range (0, count):
            self.weights.append(random.uniform(-2.5,2.5))
        
    def calculate(self, inputs):
        value = 0
        for x in range (0, len(inputs)):
            value += inputs[x]*self.weights[x]
        self.activation = Sigmoid(value)
        return self.activation
    

class Layer:
    layerNeurons = []
    
    def AssignNeurons(self, neurons):
        self.layerNeurons = neurons
        
    def CreateNeurons(self, count):
        self.layerNeurons = []
        for x in range (0, count):
            self.layerNeurons.append(Neuron())
        
    # Give Layer.Calculate a list of inputs!
    # The input should be a list of all of the previous layer's activations
    def Calculate(self, inputsPerNeuron):
        for x in range (0, len(self.layerNeurons)):
            self.layerNeurons[x].calculate(inputsPerNeuron)
            
    def CalculateFromLayer(self, otherLayer):
        inputsPerNeuron = []
        for x in range (0, len(otherLayer.layerNeurons)):
            inputsPerNeuron.append(otherLayer.layerNeurons[x].activation)
        self.Calculate(inputsPerNeuron)



inLayer = Layer()
hid1Layer = Layer()
hid2Layer = Layer()
outLayer = Layer()



# INPUT LAYER
# NOTE; DOES NOT NEED WEIGHTS
inLayer.CreateNeurons(8)


# HIDDEN LAYER 1
hid1Layer.CreateNeurons(8)
    
    
# HIDDEN LAYER 2
hid2Layer.CreateNeurons(8)
    
    
# OUTPUT LAYER
outLayer.CreateNeurons(26)


# SET WEIGHTS
for x in range (0, 8):
    hid1Layer.layerNeurons[x].RandomWeights(8)
    hid2Layer.layerNeurons[x].RandomWeights(8)
for x in range (0, 26):
    outLayer.layerNeurons[x].RandomWeights(8)




# DESIGNATE INPUT HERE
# RIGHT NOW USING MANUAL ASSIGNMENT
# INPUT SHOULD BE inLayer's ACTIVATION VALUES
inLayer.layerNeurons[0].activation = 0
inLayer.layerNeurons[1].activation = 1
inLayer.layerNeurons[2].activation = 1
inLayer.layerNeurons[3].activation = 0
inLayer.layerNeurons[4].activation = 0
inLayer.layerNeurons[5].activation = 0
inLayer.layerNeurons[6].activation = 0
inLayer.layerNeurons[7].activation = 1


# CALCULATE OUTPUT
hid1Layer.CalculateFromLayer(inLayer)
hid2Layer.CalculateFromLayer(hid1Layer)
outLayer.CalculateFromLayer(hid2Layer)


print ("====================\nINPUT LAYER ACTIVATIONS\n==================")
for x in range (0, 8):
    print (x, ":", inLayer.layerNeurons[x].activation)

print ("====================\nHIDDEN LAYER 1 ACTIVATIONS\n==================")
for x in range (0, 8):
    print (x, ":", hid1Layer.layerNeurons[x].activation)
    
print ("====================\nHIDDEN LAYER 2 ACTIVATIONS\n==================")
for x in range (0, 8):
    print (x, ":", hid2Layer.layerNeurons[x].activation)

print ("====================\nOUT LAYER ACTIVATIONS\n==================")
for x in range (0, 26):
    print (x, ":", outLayer.layerNeurons[x].activation)


# FIND HIGHEST OUTPUT
# DETERMINE ITS PREDICTION FOR THE INPUT
pos = 0
posValue = 0
for x in range (0, 26):
    if (posValue < outLayer.layerNeurons[x].activation):
        pos = x
        posValue = outLayer.layerNeurons[x].activation


print ("\n\n\n===================\nHIGHEST VALUE OUTPUT ACTIVATION:", pos)
print ("LETTER VALUE:", string.ascii_uppercase[pos])


print ("\nHello World!")


"""

for x in range (0, 8):
    hid2Layer.layerNeurons.append(Neuron())
    
    """