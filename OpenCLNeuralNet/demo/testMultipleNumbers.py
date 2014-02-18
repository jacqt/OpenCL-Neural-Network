import convert
import Image
import numpy
import time
import textwrap
import os
import pickle
import ExtractCharacters

#Reads the image located ./data/number/test.png
#Extracts the individual digits from the image
#Uses the neural net to attempt to categorize the handwritten digits

def Indicator(n):
    if n > .5: return 1
    else     : return 0

def norm(n):
    return (255-n)/255.0

def TestNumber(testVec):
    testVec = map(norm,convert.getFeatureVecFromImg(testVec))
    net.ClearNodes()
    net.SetInputs(testVec)
    rawOutputs = net.ComputeOutput()
    outputs = map (Indicator,net.ComputeOutput())
    print "Confidence Values:"
    print ("\n" + fspace).join(textwrap.wrap(fspace+str(rawOutputs).ljust(10)))
    print "Greatest Confidence:"
    print fspace + str(rawOutputs.index(max(rawOutputs))) + " with " + str(100*max(rawOutputs)) + "% confidence"
    print 
    return rawOutputs.index(max(rawOutputs))


def TestImage(path):
    testImage = Image.open(path).convert("L")
    imgArr = numpy.asarray(testImage)
    j = numpy.array(ExtractCharacters.LabelPixels(imgArr),dtype=numpy.uint8)
    imagesWithCenters = ExtractCharacters.PasteCharacters(imgArr,j)
    numbers = []
    for imageWCenter in imagesWithCenters:
        numbers.append((imageWCenter[1],TestNumber(imageWCenter[0])))
    numbers.sort()
    print "The neural net read this!"
    print "   ",
    s = ""
    for k in numbers:
         s += str(k[1])
    print s
    print

netName  = "bigassnet"
netFile = open(netName+".pkl","rb")
net = pickle.load(netFile)
print "The image we test against is located at ./data/number/test.png"
fspace = "    "
while 1:
    testFile = raw_input("Press enter to test: ")
    #time.sleep(5)
    curDir = os.getcwd()
    testFile = os.path.join(curDir,"data//number/test.png")
    TestImage(testFile)
