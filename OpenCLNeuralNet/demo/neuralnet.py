#############################################################
#                  The Basic Node Class                     #
#############################################################
import math
import random
import pickle

#Constant to determine how much to change the weights
global N
N = 0.05
global c
c = 0

class NeuralNet:
    def __init__(self):
        #An arbitrary number of layers; in reality we only use four
        self.Lists = []#list containing all the layers in the neural net
        self.rootNodes = []

    def StoreData(self):
        "Pickles itself into net_.pkl (stores the net into a file for future use)"
        global c
        output = open("net" + str(c) + ".pkl","wb")
        pickle.dump(self,output)
        output.close()
        c += 1

    def LoadData(self, fileName):
        "Loads a neural net from a file containing weights"
        netFile = open(fileName,"r")
        lines = netFile.readlines()
        #Convert every value to a float
        for i in range(1,len(lines)):
            lines[i] = lines[i].rsplit()
            for j in range(len(lines[i])):
                lines[i][j] = float(lines[i][j])

        netSpec = lines.pop(0).rsplit()
        dimensions = int(netSpec.pop(0))
        for i in range(len(netSpec)):
            netSpec[i] = int(netSpec[i])
        self.ConstructNet(dimensions,netSpec)
        for i in range(1,len(self.Lists)):
            layer = self.Lists[i]
            for j in range(len(layer)):
                node = layer[j]
                weightArray = lines.pop(0)
                for w in range(len(node.weights)):
                    node.weights[w] = weightArray[w]
        netFile.close()

    def SetInputs(self, featureVector):
        "Takes a feature vector and set them as the inputs"
        inputLayer = self.Lists[0]
        for i in range(len(inputLayer)):
            node = inputLayer[i]
            feature = featureVector[i]
            node.SetOutput(feature)

    def ClearNodes(self):
        "Sets all node outputs to None"
        for layer in self.Lists:
            for node in layer:
                node.Clear()

    def ResetNet(self):
        "Resets each node's weights"
        for layer in self.Lists:
            for node in layer:
                node.ResetWeights()

    def ComputeOutput(self):
        "Computes the output of the neural net"
        output = []
        for rootNode in self.rootNodes:
            output.append(rootNode.GetOutput())
        return output

    def TrainNet(self,featureVector, targets, constantN=None):
        "Trains the net given a feature vector and a target"
        global N
        if constantN == None:
            N = .05
        else:
            N = constantN

        self.ClearNodes()
        self.SetInputs(featureVector)
        for i in range(len(targets)):
            self.rootNodes[i].SetTarget(targets[i])

        for layer in self.Lists:
            for node in layer:
                node.GetWeightChange()
        for layer in self.Lists:
            for node in layer:
                node.ApplyWeightChange()


    def ConstructNet(self, dimensions,netSpec):
        """
        Constructs a net given the number of dimensions in a feature space
        netSpec argument is the number of nodes in each layer
            e.g. a netSpec of = [10,20,5] would yield a net with 10 nodes in layer 1
                 20 nodes in layer 2, and 5 nodes in layer 3
        i.e. the number of features we will be evaluating
        """
        #First layer is the input nodes; their values will be arbitrarily set
        newLayer = []
        for i in range(dimensions):
            newLayer.append(Node())
        self.Lists.append(newLayer)

        previousLayer = newLayer
        for i in range(len(netSpec)):
            newLayer = []
            for n in range(netSpec[i]):
                newNode = Node(previousLayer)
                newLayer.append(newNode)
            self.Lists.append(newLayer)
            previousLayer = newLayer

        self.rootNodes = self.Lists[-1]

class Node:
    def __init__(self,inputNodeList = None, defaultWeights = None):
        #We generate self.outputnodes automatically when building the neural net
        self.outputnodes = []

        self.inputs = [] #the input values to the node; we generate these automatically
        self.output = None #the output of the node
        self.delta = None #the delta variable in backprop
        self.target = None #only used for the root node; the target value we are trying to get
        self.weightchange = [] #list that stores the changes to the weight we should apply
        self.weights = [] #list of the current weights 

        if not inputNodeList:
            #A node on the output layer
            self.inputNodes = []
        elif not defaultWeights:
            #Build a set of default weights randomly for each thingy
            self.inputNodes = inputNodeList
            k = 1
            for i in range(len(inputNodeList)):
            #    if i%3 == 0:
            #        randomWeight = (.3*random.random()-.15)
            #    elif i%3 == 1:
            #        randomWeight = (.3*random.random()-.75)
            #    else:
            #        randomWeight = (.3*random.random()+.45)
                randomWeight = (.3*random.random()-.15)

                self.weights.append(randomWeight)
        else:
            #set inputNodes, weights if they are both the same length
            if (len(defaultWeights) != len(inputNodeList)):
                raise Exception("Weights passed in not the same length as inputNodeList passed in!")

            self.weights = defaultWeights
            self.inputNodes = inputNodeList

        for inputNode in self.inputNodes:
            inputNode.outputnodes.append(self)

    ############################################################# 
    #                    TRAINING CODE                          #
    ############################################################# 

    def ResetWeights(self):
        "Resets the weights to a random distribution with a gap (Not sure what the best initial weights are!)"
        length = len(self.weights)
        self.weights = []
        k = -1
        for i in range(length):
            #if i%3 == 0:
            #    randomWeight = (.3*random.random()-.15)
            #elif i%3 == 1:
            #    randomWeight = (.3*random.random()-.105)
            #else:
            #    randomWeight = (.3*random.random()+.75)
            randomWeight = (.3*random.random()-.15)
            self.weights.append(randomWeight)

    def SetTarget(self,target):
        "Sets the target of the node (only used for the root node!)"
        self.target = target

    def ComputeWeightChange(self):
        "Computes the change in weights"
        if self.weightchange != []:
            return False

        for i in range(len(self.inputNodes)):
            inputNode = self.inputNodes[i]
            a = inputNode.GetOutput()
            d = self.GetDelta()
            weightChange = N*d*a
            self.weightchange.append(weightChange)

    def GetWeightChange(self):
        "Returns the change in weights, or computes it first and then returns it"
        if self.weightchange == []:
            self.ComputeWeightChange()
        return self.weightchange

    def ApplyWeightChange(self):
        "Applies the change in weights, or computes it first and then applies it"
        if self.weightchange == []:
            self.ComputeWeightChange()
        for i in range(len(self.weightchange)):
            weightChange = self.weightchange[i]
            self.weights[i] += weightChange
        #WHAT IS GOING ON BELOW?????
        g = 0
        for k in self.weightchange:
            g += k

    def ComputeDelta(self):
        "Computes the delta value for backprop"
        if self.delta != None:
            return False
        if self.output == None:
            self.ComputeOutput()

        u = Node.SigmoidDerivative(self.output)

        if self.outputnodes == []:
            #the output node!
            t = self.target
            a = self.GetOutput()
            self.delta = u*(t-a)
        else:
            recurse = 0
            for i in range(len(self.outputnodes)):
                node = self.outputnodes[i]
                index = node.inputNodes.index(self)
                w = node.weights[index]
                delta = node.GetDelta()
                recurse += delta*w
            self.delta = u*recurse

    def GetDelta(self):
        "Returns the delta value, or computes the value and then returns it"
        if self.delta == None:
            self.ComputeDelta()
        return self.delta

    ############################################################# 
    #                 CALCULATE OUTPUT CODE                     #
    ############################################################# 

    def ComputeOutput(self):
        "Computes the output of the node"
        if self.output != None:
            return False 

        if self.inputs == []:
            self.GetInput()

        output = 0
        for i in range(len(self.inputs)):
            val = self.inputs[i]
            weight = self.weights[i]
            output += val*weight
        
        self.output = output
        return True

    def SetOutput(self,arbitraryOutput):
        "Arbitrarily sets the output of the node; used to set the inputs to the neural net"
        self.output = arbitraryOutput

    def Clear(self):
        "Clears the output, input, delta, weightchange, and target attributes"
        self.inputs = []
        self.output = None
        self.delta = None
        self.target = None
        self.weightchange = []

    def GetInput(self):
        "Calls GetOutput of all the feeder nodes and builds a vector out of them"
        for inputNode in self.inputNodes:
            self.inputs.append(inputNode.GetOutput())
        pass

    def GetOutput(self):
        "Returns the output of the current node, or computes it"
        if self.output == None:
            self.ComputeOutput()

        if self.inputNodes == []:
            return self.output # if it is an input node then doing do a sigmoid on it!
        return self.Sigmoid(self.output)
 
    ############################################################# 
    #                USEFUL STATIC FUNCTIONS                    #
    #############################################################

    @staticmethod
    def Sigmoid(n):
        "Returns the sigmoid function"
        if n < -200:
            return 0
        if n > 200:
            return 1
        return 1/(1+math.e**(-n))

    @staticmethod
    def SigmoidDerivative(n):
        "Returns the derivative of the sigmoid function"
        return Node.Sigmoid(n)*(1 - Node.Sigmoid(n))

def LoadNet():
    "Loads a net from net.pkl"
    inputFile = open("net.pkl","rb")
    newNet = pickle.load(inputFile)
    return newNet
