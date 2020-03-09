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
import json

# given a value, returns it in a range from 0 to 1.
# 0 is 0.5, 5 is close to 1, -5 is close to 0.
def Sigmoid (input):
    output = 1/(1 + math.exp(-input)) 
    return output

class Neuron:
    
    def __init__ (self, name):
        self.name = name
        self.weights = []
        self.activation = 0.0
        self.minMutate = -2.5
        self.maxMutate = 2.5

    # resets all weights, then randomly generates "count" new weights from min and max mutation rates
    def RandomWeights(self, count):
        self.weights = []
        #print (self.name, len(self.weights), count)
        for x in range (0, count):
            self.weights.append(random.uniform(self.minMutate,self.maxMutate))
        #print (len(self.weights))

    def Calculate(self, inputs):
        if len(inputs) != len(self.weights):
            #print("ERROR; INCORRECT NEURON INPUT SIZE:", self.name, "\t INPUTvWEIGHT SIZE:", len(inputs), len(self.weights))
            return -999
        else:
            activation = 0
            for x in range (0, len(inputs)):
                activation += inputs[x]*self.weights[x]
            self.activation = Sigmoid(activation)
            return self.activation

    def MutateNeuron (self, min, max):
        #print ("Mutating Neuron", self.name, "and it's", len(self.weights),"weights.")
        for x in range (0, len(self.weights)):
            self.weights[x] = self.weights[x] * random.uniform(min, max)

class Layer:
    
    def __init__ (self):
        self.layerNeurons = []
        self.name = "layer?"
    
    def AssignNeurons(self, neurons):
        self.layerNeurons = neurons
        
    def CreateNeurons(self, count, name):
        self.layerNeurons = []
        #print (self.name, count)
        for x in range (0, count):
            self.layerNeurons.append(Neuron(name))
        
    # Give Layer.Calculate a list of inputs!
    # The input should be a list of all of the previous layer's activations
    def Calculate(self, inputsPerNeuron):
        for x in range (0, len(self.layerNeurons)):
            #print (inputsPerNeuron, self.layerNeurons[x].weights)
            self.layerNeurons[x].Calculate(inputsPerNeuron)
            
    def CalculateFromLayer(self, otherLayer):
        inputsPerNeuron = []
        for x in range (0, len(otherLayer.layerNeurons)):
            inputsPerNeuron.append(otherLayer.layerNeurons[x].activation)
        self.Calculate(inputsPerNeuron)
        
    def Mutate (self, min, max):
        #print ("Mutating", len(self.layerNeurons), "times.")
        for x in range (0, len(self.layerNeurons)):
            self.layerNeurons[x].MutateNeuron(min, max)

    # for each neuron, Randomize Weights passing the amount of weights needed
    def RandomizeNeuronWeights (self, weightCount):
        #print ("PASSING", weightCount)
        for x in range (0, len(self.layerNeurons)):
            self.layerNeurons[x].RandomWeights(weightCount)

########################################################################################################################################################
########################################################################################################################################################
#^ LAYER CLASS
#
#v ANN CLASS
########################################################################################################################################################
########################################################################################################################################################

# ADD HASH TABLE FOR LETTER BINARY THING
# TIDY UP TO MAKE MORE USER FRIENDLY
# RIGHT NOW MANUALLY SETTING A LOT OF THINGS
class ANN:
    
    def __init__ (self):
        self.alphabet = {'A': LetterToBinary(string.ascii_uppercase[0])}
        self.layers = []        # UNUSED
        self.accuracy = 0.0
        self.accuracyCount = 0
        self.inLayer = Layer()
        self.hid1Layer = Layer()
        self.hid2Layer = Layer()
        self.outLayer = Layer()
        self.InitializeLayers(10, 2)
        self.InitializeRandom()

    def SetParent (self, parent):
        self.hid1Layer = copy.deepcopy(parent.hid1Layer)
        self.hid2Layer = copy.deepcopy(parent.hid2Layer)
        self.outLayer = copy.deepcopy(parent.outLayer)

    # THIS WORKS
    def MutateANN (self, min, max):
        #self.inLayer.Mutate()          # DON'T NEED TO MUTATE INPUT BC IT HAS NO WEIGHTS
        #print ("MUTATING B")
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
        self.InitializeLayers(10, 2)

    # To clarify; neuronsPerLayer is how many neurons you want per HIDDEN layer.
    def InitializeLayers (self, neuronsPerLayer, hiddenLayers):

        # INPUT LAYER
        # NOTE; DOES NOT NEED WEIGHTS
        self.inLayer.CreateNeurons(8, "in")

        self.layers = []
        for x in range (0, hiddenLayers):
            self.layers.append(Layer())
            if x == 0:
                #print ("hid0 being printed.", ("hid" + str(x)))
                self.layers[x].CreateNeurons(8, ("hid" + str(x)))
                self.layers[x].name = ("hid" + str(x))
            else:
                #print ("hid1 being printed.", ("hid" + str(x)), neuronsPerLayer)
                self.layers[x].CreateNeurons(neuronsPerLayer, ("hid" + str(x)))
                self.layers[x].name = ("hid" + str(x))
        #print (self.layers)

        # HIDDEN LAYER 1
        self.hid1Layer = self.layers[0]           
        #self.hid1Layer.CreateNeurons(neuronsPerLayer, "hid1")           

        # HIDDEN LAYER 2
        self.hid2Layer = self.layers[1]
        #self.hid2Layer.CreateNeurons(neuronsPerLayer, "hid2")            

        # OUTPUT LAYER
        self.outLayer.CreateNeurons(26, "out")    

    # PASSING 8 FOR HID1/second hidden layer when  should be passing 10
    def InitializeRandom (self):
        # SET RANDOM WEIGHTS
        #print ("AAAAAAAA THIS SHOULD BE 8: ", len(self.hid1Layer.layerNeurons))
        self.hid1Layer.RandomizeNeuronWeights(len(self.hid1Layer.layerNeurons))
        #print (len(self.hid2Layer.layerNeurons))
        self.hid2Layer.RandomizeNeuronWeights(len(self.hid2Layer.layerNeurons))
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
        #print (letter)
        #if letter == "B":
            #print (self.alphabet)
        binary = self.alphabet[letter]
        #print (binary)
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
        self.accuracyCount = 0
        #print (self.hid1Layer.layerNeurons[0].activation)
        for x in range (0, len(letters)):
            self.InputLetter(letters[x])
            self.ANNCalculate()       
            if (verbose):
                print ("TEST ",x,":   GIVEN[",letters[x],"], PREDICTED[",self.GetPrediction(),"]", sep='')
            correct += self.CheckGuess (letters[x], self.GetPrediction())
            if (letters[x] == self.GetPrediction()):
                self.accuracyCount += 1
        if (verbose):        
            print ("\nIt got", correct, "correct.")
            print ("====================\nACCURACY:",100 * (correct/len(letters)),"\n====================")
        return (correct/len(letters))

    def CheckGuess (self, correct, prediction):
        binaryCorrect = ' '.join(format(ord(x), 'b') for x in correct)
        binaryPrediction = ' '.join(format(ord(x), 'b') for x in prediction)
        counter = -4
        
        for x in range (0, len(binaryCorrect)):
            if (binaryCorrect[x] == binaryPrediction[x]):
                counter += 1
        
        return Sigmoid(counter)

    def Prettify (self):
        sendOut = "ANN WITH HID1[0].WEIGHT[0] OF: "
        sendOut += str(self.hid1Layer.layerNeurons[0].weights[0])
        #self.Display()
        return sendOut

class ANNManager:
    
    def __init__ (self):
        anns = []
        
        # create the alphabet
        self.alphabet = {'A': LetterToBinary(string.ascii_uppercase[0])}
        for x in range (0, 26):
            self.alphabet[string.ascii_uppercase[x]] = LetterToBinary(string.ascii_uppercase[x])

    # CREATES NEW BATCH OF ANNS OF SIZE BATCHSIZE
    def RunANNBatch (self, batchSize):
        annieBall = []  
        for x in range (0, batchSize):
            annieBall.append(ANN())
            annieBall[x].alphabet = self.alphabet
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
        return annieBall

    # JUST RUNS A BUNCH OF ANNS    
    def RunANN (self, anns):
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

    # JUST GET ACCURACY DONT RUN THE WHOLE THING OVER AGAIN PLEASE
    def DisplayANNs (self,anns):
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
        print ("Average Accuracy:", round(100*avgAcc,3), "\t\tMost Accurate:", round(100*mostAcc,3), "at", mostAccPos, "\twith", anns[x].accuracyCount, "out of 26 correct.")

    # SORT BY DOING THIS
    # TWO POINTERS, ONE AT START, TWO IMMEDIATELY AFTER ONE
    # COMPARE ONE AND TWO. IF TWO IS GREATER, SWAP, OTHERWISE, MOVE TWO UP BY ONE
    # REPEAT UNTIL TWO REACHES THE END OF THE LIST
    # REPEAT, BUT WITH ONE MOVED UP BY ONE.
    # REPEAT UNTIL ONE IS THE LAST ELEMENT IN THE LIST
    def SortANNs (self, annArray):
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
    def CullANNs (self, anns, survivalPercentage):
        survivors = []
        for x in range (0, round(len(anns)*survivalPercentage)):
            survivors.append(anns[x])
            anns[x].alphabet = self.alphabet
        return survivors

    def RepopulateANNs (self, anns, survivalPercentage, randomPercentage):
        population = []
        currentParent = 0
        for x in range (0, len (anns)):
            # for those between 0 and the survival percentage, add them back in. They made it.
            if (x < round(len(anns)*survivalPercentage)):
                population.append(anns[x])
            # for those between the survival percentage and the random percentage, add in new randoms.
            elif (round(len(anns)*survivalPercentage) < x and x < round(len(anns)*randomPercentage) + round(len(anns)*survivalPercentage)):
                temp = ANN()
                temp.alphabet = self.alphabet
                population.append(temp)
            # otherwise, breed new children!
            else:
                temp = copy.deepcopy(anns[currentParent])
                #print ("MUTATING")
                temp.MutateANN(0.75,1.25)
                temp.alphabet = self.alphabet
                population.append(temp)
                currentParent += 1
                if (currentParent >= round(len(anns)*survivalPercentage)):
                    currentParent = 0
        return population


def LetterToBinary (letter):
    # LINE I STOLE FROM ONLINE THAT CONVERTS THE LETTER INTO ITS BINARY FORM IN A STRING
    binary = ' '.join(format(ord(x), 'b') for x in letter)
    outBin = [0.0]
    # CONVERT THE STRING TO AN ARRAY OF FLOATS
    for x in range (0, 7):
        outBin.append(float(binary[x]))
    return outBin


# create the manager
manager = ANNManager()

#AyyNNs = RunANNBatch(12)
AyyNNs = manager.RunANNBatch(12)

manager.SortANNs(AyyNNs)
AyyNNs = manager.RepopulateANNs (AyyNNs, 0.3,0.3)
#RunANN(AyyNNs)
manager.RunANN(AyyNNs)

manager.SortANNs(AyyNNs)
#for x in range (0, 12):
    #print (x, ": ", AyyNNs[x].accuracy, sep="")

for y in range (0, 1000):
    manager.SortANNs(AyyNNs)
    AyyNNs = manager.RepopulateANNs (AyyNNs, 0.3,0.3)
    manager.RunANN(AyyNNs)
    if (y % 10 == 0):
        manager.DisplayANNs(AyyNNs)
    
ANNManager()

print ("\nHello World!")