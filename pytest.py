# The network will take in a letter convereted to binary.
# It will try to guess the letter. If it guesses correctly, then yay!
# Otherwise, boo.

# INPUT - 8 NEURONS OF 1 OR 0
# TWO HIDDEN LAYERS OF 8 EACH
# OUTPUT - 26 NEURONS, ONE FOR EACH LETTER

import math
import random
import string
import binascii

def Sigmoid (input):
    output = 1/(1 + math.exp(-input)) 
    return output

class Neuron:
    activation = 0.0
    
    def __init__ (self):
        self.weights = []
    
    def SetWeights(self, inWeights):
        self.weights = inWeights
        
    def RandomWeights(self, count):
        self.weights = []
        for x in range (0, count):
            self.weights.append(random.uniform(-2.5,2.5))
        
    def calculate(self, inputs):
        #print("CALCULATING")
        value = 0
        for x in range (0, len(inputs)):
            value += inputs[x]*self.weights[x]
        self.activation = Sigmoid(value)
        return self.activation
        
    def Mutate (self):
        for x in range (0, len(self.weights)):
            self.weights[x] = self.weights[x] * random.uniform(-10.1, 10.1)
    

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
        
    def Mutate (self):
        for x in range (0, len(self.layerNeurons)):
            self.layerNeurons[x].Mutate()


class ANN:
    
    def __init__ (self):
        self.layers = []
        self.inLayer = Layer()
        self.hid1Layer = Layer()
        self.hid2Layer = Layer()
        self.outLayer = Layer()
        self.InitializeLayers()
        self.InitializeRandom()
    
    def SetParent (self, parent):
        self.Clear()
        self.layers = []
        self.inLayer.layerNeurons = parent.inLayer.layerNeurons.copy()
        self.hid1Layer.layerNeurons = parent.hid1Layer.layerNeurons.copy()
        self.hid2Layer.layerNeurons = parent.hid2Layer.layerNeurons.copy()
        self.outLayer.layerNeurons = parent.outLayer.layerNeurons.copy()
        self.MutateANN()
        
    def MutateANN (self):
        self.inLayer.Mutate()
        self.hid1Layer.Mutate()
        self.hid2Layer.Mutate()
        self.outLayer.Mutate()
    
    # WARNING; DOES NOT REPLACE LAYERS ONCE CLEARING.
    # CLEARING AND CALCULATING WILL RETURN AN ERROR.
    def Clear (self):
        self.layers = []
        self.inLayer = Layer()
        self.hid1Layer = Layer()
        self.hid2Layer = Layer()
        self.outLayer = Layer()
    
    def InitializeLayers (self):
        # INPUT LAYER
        # NOTE; DOES NOT NEED WEIGHTS
        self.inLayer.CreateNeurons(8)
        # HIDDEN LAYER 1
        self.hid1Layer.CreateNeurons(8)           
        # HIDDEN LAYER 2
        self.hid2Layer.CreateNeurons(8)            
        # OUTPUT LAYER
        self.outLayer.CreateNeurons(26)    
    
    def InitializeRandom (self):
        # SET RANDOM WEIGHTS
        for x in range (0, 8):
            self.hid1Layer.layerNeurons[x].RandomWeights(8)
            self.hid2Layer.layerNeurons[x].RandomWeights(8)
        for x in range (0, 26):
            self.outLayer.layerNeurons[x].RandomWeights(8)   
    
    def Display (self):
        print ("====================\nINPUT LAYER ACTIVATIONS\n==================")
        for x in range (0, 8):
            print (x, ":", self.inLayer.layerNeurons[x].activation)
        print ("====================\nHIDDEN LAYER 1 ACTIVATIONS\n==================")
        for x in range (0, 8):
            print (x, ":", self.hid1Layer.layerNeurons[x].activation)   
        print ("====================\nHIDDEN LAYER 2 ACTIVATIONS\n==================")
        for x in range (0, 8):
            print (x, ":", self.hid2Layer.layerNeurons[x].activation)
        print ("====================\nOUT LAYER ACTIVATIONS\n==================")
        for x in range (0, 26):
            print (x, ":", self.outLayer.layerNeurons[x].activation)
        # FIND HIGHEST OUTPUT
        # DETERMINE ITS PREDICTION FOR THE INPUT
        pos = 0
        posValue = 0
        for x in range (0, 26):
            if (posValue < self.outLayer.layerNeurons[x].activation):
                pos = x
                posValue = self.outLayer.layerNeurons[x].activation
        print ("\n\n\n===================\nHIGHEST VALUE OUTPUT ACTIVATION:", pos)
        print ("LETTER VALUE:", string.ascii_uppercase[pos])
    
    # RETURNS THE PREDICTED LETTER OF THE CURRENT INPUT THE ANN HAS
    def GetPrediction (self):
        # FIND HIGHEST OUTPUT
        # DETERMINE ITS PREDICTION FOR THE INPUT
        pos = 0
        posValue = 0
        for x in range (0, 26):
            if (posValue < self.outLayer.layerNeurons[x].activation):
                pos = x
                posValue = self.outLayer.layerNeurons[x].activation    
        return string.ascii_uppercase[pos]
    
    # TURN LETTER TO BINARY ARRAY, FEED INTO INPUT ARRAY
    def InputLetter (self, letter):
        # LINE I STOLE FROM ONLINE THAT CONVERTS THE LETTER INTO ITS BINARY FORM IN A STRING
        binary = ' '.join(format(ord(x), 'b') for x in letter)
        outBin = [0.0]
        # CONVERT THE STRING TO AN ARRAY OF FLOATS
        for x in range (0, 7):
            outBin.append(float(binary[x]))
        # FILL INPUT ARRAY
        for x in range (0, 8):
            self.inLayer.layerNeurons[x].activation = outBin[x]
    
    # CALCULATE ANN'S LAYERS
    def ANNCalculate (self):
        self.hid1Layer.CalculateFromLayer(self.inLayer)
        self.hid2Layer.CalculateFromLayer(self.hid1Layer)
        self.outLayer.CalculateFromLayer(self.hid2Layer)    
    
    # TAKE LIST OF LETTERS, RUN THROUGH FOR EACH LETTER.
    def InputLetterList (self, letters, verbose):
        correct = 0
        for x in range (0, len(letters)):
            self.InputLetter(letters[x])
            self.ANNCalculate()       
            if (verbose):
                print ("TEST ",x,":   GIVEN[",letters[x],"], PREDICTED[",self.GetPrediction(),"]", sep='')
            if (letters[x] == self.GetPrediction()):
                correct += 1
        if (verbose):        
            print ("\nIt got", correct, "correct.")
            print ("====================\nACCURACY:",100 * (correct/len(letters)),"\n====================")
        return (correct/len(letters))






#annie = ANN()
#annie.InputLetter('B')
#annie.InputLetter('C')
#annie.Display()
#annie.InputLetterList(['A','B','C'])
#annie.InputLetterList(string.ascii_uppercase)

annieBall = []

for x in range (0, 10):
    annieBall.append(ANN())

acc = 0
avgAcc = 0
mostAcc = 0
mostAccPos = 0

annieBall[9].SetParent(annieBall[8])

for x in range (0, 10):
    acc = annieBall[x].InputLetterList(string.ascii_uppercase, True)
    avgAcc += acc
    if (acc > mostAcc):
        mostAcc = acc
        mostAccPos = x
        
avgAcc = avgAcc/10

print ("Average Accuracy:", 100*avgAcc)
print ("Most Accurate:", 100*mostAcc, "at", mostAccPos)






print ("\nHello World!")