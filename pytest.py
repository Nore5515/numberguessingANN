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
import copy

def Sigmoid (input):
    output = 1/(1 + math.exp(-input)) 
    return output

class Neuron:
    
    def __init__ (self):
        self.weights = []
        self.activation = 0.0
    
    def SetWeights(self, inWeights):
        self.weights = inWeights
        
    def RandomWeights(self, count):
        self.weights = []
        for x in range (0, count):
            self.weights.append(random.uniform(-2.5,2.5))
        #print ("Reset Weights, generating new ones.", self.weights[0])
        
    def calculate(self, inputs):
        #print("CALCULATING")
        value = 0
        for x in range (0, len(inputs)):
            value += inputs[x]*self.weights[x]
        self.activation = Sigmoid(value)
        return self.activation
        
    def MutateNeuron (self, min, max):
        for x in range (0, len(self.weights)):
            self.weights[x] = self.weights[x] * random.uniform(min, max)
    

class Layer:
    
    def __init__ (self):
        self.layerNeurons = []
    
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
        
    def Mutate (self, min, max):
        for x in range (0, len(self.layerNeurons)):
            self.layerNeurons[x].MutateNeuron(min, max)

    # for each neuron, Randomize Weights passing the amount of weights needed
    def RandomizeNeuronWeights (self, weightCount):
        for x in range (0, len(self.layerNeurons)):
            self.layerNeurons[x].RandomWeights(weightCount)


########################################################################################################################################################
########################################################################################################################################################
#^ LAYER CLASS
#
#v ANN CLASS
########################################################################################################################################################
########################################################################################################################################################

# TIDY UP TO MAKE MORE USER FRIENDLY
# RIGHT NOW MANUALLY SETTING A LOT OF THINGS
class ANN:
    
    def __init__ (self):
        self.layers = []
        self.accuracy = 0.0
        self.accuracyCount = 0
        self.inLayer = Layer()
        self.hid1Layer = Layer()
        self.hid2Layer = Layer()
        self.outLayer = Layer()
        self.InitializeLayers(10)
        self.InitializeRandom()
    
    def SetParent (self, parent):
        self.hid1Layer = copy.deepcopy(parent.hid1Layer)
        self.hid2Layer = copy.deepcopy(parent.hid2Layer)
        self.outLayer = copy.deepcopy(parent.outLayer)
    
    # THIS WORKS
    def MutateANN (self, min, max):
        #self.inLayer.Mutate()          # DON'T NEED TO MUTATE INPUT BC IT HAS NO WEIGHTS
        self.hid1Layer.Mutate(min, max)
        self.hid2Layer.Mutate(min, max)
        self.outLayer.Mutate(min, max)
    
    # WARNING; DOES NOT REPLACE LAYERS ONCE CLEARING.
    # CLEARING AND CALCULATING WILL RETURN AN ERROR.
    def Clear (self):
        self.layers = []
        self.inLayer = Layer()
        self.hid1Layer = Layer()
        self.hid2Layer = Layer()
        self.outLayer = Layer()
        self.InitializeLayers(10)
    
    def InitializeLayers (self, neuronsPerLayer):
        # INPUT LAYER
        # NOTE; DOES NOT NEED WEIGHTS
        self.inLayer.CreateNeurons(8)
        # HIDDEN LAYER 1
        self.hid1Layer.CreateNeurons(neuronsPerLayer)           
        # HIDDEN LAYER 2
        self.hid2Layer.CreateNeurons(neuronsPerLayer)            
        # OUTPUT LAYER
        self.outLayer.CreateNeurons(26)    
    
    def InitializeRandom (self):
        # SET RANDOM WEIGHTS
        self.hid1Layer.RandomizeNeuronWeights(len(self.hid1Layer.layerNeurons))
        self.hid2Layer.RandomizeNeuronWeights(len(self.hid1Layer.layerNeurons))
        self.outLayer.RandomizeNeuronWeights(26)
    
    def Display (self):
        print ("====================\nINPUT LAYER ACTIVATIONS\n==================")
        for x in range (0, len(self.inLayer.layerNeurons)):
            print (x, ":", round(self.inLayer.layerNeurons[x].activation,3))
        print ("====================\nHIDDEN LAYER 1 ACTIVATIONS\n==================")
        for x in range (0, len(self.hid1Layer.layerNeurons)):
            print (x, ":",  round(self.hid1Layer.layerNeurons[x].activation,3))   
        print ("====================\nHIDDEN LAYER 2 ACTIVATIONS\n==================")
        for x in range (0, len(self.hid2Layer.layerNeurons)):
            print (x, ":",  round(self.hid2Layer.layerNeurons[x].activation,3))
        print ("====================\nOUT LAYER ACTIVATIONS\n==================")
        for x in range (0, 26):
            print (x, ":",  round(self.outLayer.layerNeurons[x].activation,3))
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
        #print (self.hid1Layer.layerNeurons[0].activation)
        for x in range (0, len(letters)):
            self.InputLetter(letters[x])
            self.ANNCalculate()       
            if (verbose):
                print ("TEST ",x,":   GIVEN[",letters[x],"], PREDICTED[",self.GetPrediction(),"]", sep='')
            correct += self.CheckGuess (letters[x], self.GetPrediction())
            #if (letters[x] == self.GetPrediction()):
                #correct += CheckGuess(
        if (verbose):        
            print ("\nIt got", correct, "correct.")
            print ("====================\nACCURACY:",100 * (correct/len(letters)),"\n====================")
        return (correct/len(letters))

    def CheckGuess (self, correct, prediction):
        binaryCorrect = ' '.join(format(ord(x), 'b') for x in correct)
        binaryPrediction = ' '.join(format(ord(x), 'b') for x in prediction)
        counter = 0
        
        for x in range (0, len(binaryCorrect)):
            if (binaryCorrect[x] == binaryPrediction[x]):
                counter += 1
        
        return counter / len(binaryCorrect)

    def Prettify (self):
        sendOut = "ANN WITH HID1[0].WEIGHT[0] OF: "
        sendOut += str(self.hid1Layer.layerNeurons[0].weights[0])
        #self.Display()
        return sendOut
        
    #def CorrectCount (self):
        



class ANNManager:
    
    def __init__ (self):
        anns = []


# CREATES NEW BATCH OF ANNS OF SIZE BATCHSIZE
def RunANNBatch (batchSize):
    annieBall = []  
    for x in range (0, batchSize):
        annieBall.append(ANN())
    acc = 0
    avgAcc = 0
    mostAcc = 0
    mostAccPos = 0    
    for x in range (0, len(annieBall)):
        acc = annieBall[x].InputLetterList(string.ascii_uppercase, False)
        annieBall[x].accuracy = acc
        avgAcc += acc
        if (acc > mostAcc):
            mostAcc = acc
            mostAccPos = x 
    avgAcc = avgAcc/len(annieBall)
    #print ("Average Accuracy:", 100*avgAcc)
    #print ("Most Accurate:", 100*mostAcc, "at", mostAccPos)
    return annieBall
    

# JUST RUNS A BUNCH OF ANNS    
def RunANN (anns):
    acc = 0
    avgAcc = 0
    mostAcc = 0
    mostAccPos = 0    
    for x in range (0, 10):
        acc = anns[x].InputLetterList(string.ascii_uppercase, False)
        anns[x].accuracy = acc 
        avgAcc += acc
        if (acc > mostAcc):
            mostAcc = acc
            mostAccPos = x 
    avgAcc = avgAcc/10
    #print ("Average Accuracy:", 100*avgAcc)
    #print ("Most Accurate:", 100*mostAcc, "at", mostAccPos)

# JUST GET ACCURACY DONT RUN THE WHOLE THING OVER AGAIN PLEASE
def DisplayANNs (anns):
    acc = 0
    avgAcc = 0
    mostAcc = 0
    mostAccPos = 0    
    for x in range (0, len(anns)):
        acc = anns[x].InputLetterList(string.ascii_uppercase, False)
        anns[x].accuracy = acc
        avgAcc += acc
        if (acc > mostAcc):
            mostAcc = acc
            mostAccPos = x 
    avgAcc = avgAcc/len(anns)
    print ("Average Accuracy:", 100*avgAcc, "\t\tMost Accurate:", 100*mostAcc, "at", mostAccPos)

# SORT BY DOING THIS
# TWO POINTERS, ONE AT START, TWO IMMEDIATELY AFTER ONE
# COMPARE ONE AND TWO. IF TWO IS GREATER, SWAP, OTHERWISE, MOVE TWO UP BY ONE
# REPEAT UNTIL TWO REACHES THE END OF THE LIST
# REPEAT, BUT WITH ONE MOVED UP BY ONE.
# REPEAT UNTIL ONE IS THE LAST ELEMENT IN THE LIST
def SortANNs (annArray):
    one, two = 0, 1
    temp = ANN()
    while one < len(annArray):
        while two < len(annArray):
            if annArray[one].accuracy < annArray[two].accuracy:
                temp = copy.deepcopy(annArray[one])
                annArray[one] = annArray[two]
                annArray[two] = temp
            two += 1
        one += 1
        two = one + 1
    
# ONLY GIVE SORTED LISTS
# make this a number between 0 and 1
def CullANNs (anns, survivalPercentage):
    survivors = []
    for x in range (0, round(len(anns)*survivalPercentage)):
        survivors.append(anns[x])
    return survivors

def RepopulateANNs (anns, size):
    population = []
    counter = 0
    temp = ANN()
    for x in range (0, len(anns)):
        population.append(anns[x])
    for x in range (0, size - len(anns)):
        #population.append(ANN())
        temp = copy.deepcopy(anns[counter])
        temp.MutateANN(0.8,1.2)
        counter += 1
        if (counter >= len(anns)):
            counter = 0
            temp.MutateANN (0.8,1.2)
        population.append(temp)
    return population


AyyNNs = RunANNBatch(12)

SortANNs(AyyNNs)
#for x in range (0, 12):
    #print (x, ": ", AyyNNs[x].accuracy, sep="")

AyyNNs = CullANNs (AyyNNs, 0.25)
#print (len(AyyNNs))
AyyNNs = RepopulateANNs (AyyNNs, 12)
#print ("\n\n")
RunANN(AyyNNs)

SortANNs(AyyNNs)
#for x in range (0, 12):
    #print (x, ": ", AyyNNs[x].accuracy, sep="")

for y in range (0, 1000):
    SortANNs(AyyNNs)
    AyyNNs = CullANNs (AyyNNs, 0.3)
    AyyNNs = RepopulateANNs (AyyNNs, 12)
    RunANN(AyyNNs)
    if (y % 10 == 0):
        DisplayANNs(AyyNNs)
    
    
#DisplayANNs(AyyNNs)
ANNManager()

for x in range (0, len (AyyNNs)):
    print (AyyNNs[x].Prettify())



"""
#annie = ANN()
#annie.InputLetter('B')
#annie.InputLetter('C')
#annie.Display()
#annie.InputLetterList(['A','B','C'])
#annie.InputLetterList(string.ascii_uppercase)
# A GOOD EXAMPLE ON MUTATION WORKING AND STUFF, USE DEEPCOPY TO COPY CHILDREN
Ann = ANN()
Papa = ANN()
print ("",Ann.hid1Layer.layerNeurons[0].weights[0])
print ("",Papa.hid1Layer.layerNeurons[0].weights[0])
Ann.SetParent(Papa)
print ("\n",Ann.hid1Layer.layerNeurons[0].weights[0])
print ("",Papa.hid1Layer.layerNeurons[0].weights[0])
Ann.MutateANN()
print ("\n",Ann.hid1Layer.layerNeurons[0].weights[0])
print ("",Papa.hid1Layer.layerNeurons[0].weights[0])
"""

print ("\nHello World!")