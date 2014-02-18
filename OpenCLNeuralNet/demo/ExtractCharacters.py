import os
import Image
import numpy
import convert
from copy import deepcopy

#Extracts the individual characters from a file assuming that there are no
#overlaps in the characters

WHITE = 255

def GetNorth(imgArr,coord):
    "Returns the value of northern pixel"
    row = coord[0]-1
    col = coord[1]
    if row < 0:
        return WHITE #Assume that the northern edge is all white pixels
    else:
        return imgArr[row][col]

def GetWest(imgArr,coord):
    "Returns the value of the western pixel"
    row = coord[0]
    col = coord[1]-1
    if col < 0:
        return WHITE #Assume that the western edge is all white pixels
    else:
        return imgArr[row][col]

def GetNorthLabels(labels,coord):
    "Returns the label of the northern pixel"
    if coord[0] == 0:
        return 0
    else:
        return labels[coord[0]-1][coord[1]]

def GetWestLabels(labels,coord):
    "Returns the label of the western pixel"
    if coord[1] == 0:
        return 0
    else:
        return labels[coord[0]][coord[1]-1]

def Equiv(p1,p2):
    if p1 == p2:
        return True
    if p1 < WHITE and p2 < WHITE:
        return True
    return False

def LabelPixels(imgArr):
    """
    Takes an image array and returns a list of the same dimensions/size with labels, 
        0 being whitespace, and all other values being particular classes
    We consider two pixels to be the same if the pixels are:
        - Both white
        - Both not white
    """
    #Two pass algorithm
    tempLabels = imgArr.tolist() #create list of the same size as the numpy arr
    tempLabels = map(lambda row: map(lambda x: 0,row), tempLabels)
    equivalences = {0:[0]}
    classCounter = 0
    for i in range(len(imgArr)):
        row = imgArr[i]
        for j in range(len(row)):
            pixel = row[j]
            northPixel = GetNorth(imgArr,(i,j))
            northLabel = GetNorthLabels(tempLabels,(i,j))
            westPixel = GetWest(imgArr,(i,j))
            westLabel = GetWestLabels(tempLabels,(i,j))

            if pixel == WHITE:
                continue

            if (not Equiv(northPixel, pixel)) and (not Equiv(westPixel, pixel)):
                classCounter += 1
                tempLabels[i][j] = classCounter
                equivalences[classCounter] = [classCounter]
            else:
                if Equiv(northPixel,pixel) and Equiv(northPixel, westPixel) and northLabel != westLabel:
                    equivalences[northLabel].append(westLabel)
                    equivalences[westLabel].append(northLabel)
                    ma = max([westLabel,northLabel])
                    mi = min([westLabel,northLabel])
                    tempLabels[i][j] = mi
                    for a in range(len(tempLabels)):
                        rowd = tempLabels[i]
                        for b in range(len(rowd)):
                            if rowd[b] == ma: rowd[b] = mi

                elif Equiv(pixel, westPixel):
                    tempLabels[i][j] = westLabel

                elif Equiv(pixel, northPixel):
                    tempLabels[i][j] = northLabel

    for i in range(len(tempLabels)):
        row = tempLabels[i]
        for j in range(len(row)):
            z = row[j]
            while z > min(equivalences[z]):
                z = min(equivalences[z])
            row[j] = z# min(equivalences[row[j]])
    k = []
    for i in range(len(tempLabels)):
        row = tempLabels[i]
        for j in range(len(row)):
            if row[j] in k: continue
            else: k.append(row[j])
    newEquiv = {}
    for i in range(len(k)):
        newEquiv[k[i]] = i

    for i in range(len(tempLabels)):
        row = tempLabels[i]
        for j in range(len(row)):
            row[j] = newEquiv[row[j]]
    return tempLabels

def GetDimensions(coords):
    top = coords[0][0]
    bottom = coords[0][0]
    left = coords[0][1]
    right = coords[0][1]
    for p in coords:
        if p[0]  < top: 
            top = p[0]
        elif p[0] > bottom:
            bottom = p[0]
        if p[1] < left:
            left = p[1]
        elif p[1] > right:
            right = p[1]
    return (bottom-top, right-left)
g = 0
def CreateImage(imgArr,coords):
    global g
    dim = GetDimensions(coords)
    maxim = int(max(dim)*1.50)
    print "dims", dim, maxim
    newImg = numpy.zeros((maxim,maxim),dtype=numpy.uint8)
    newImg = map(lambda row: map(lambda e: 255,row),newImg)
    center = convert.getCenter(coords)
    if maxim%2 == 0:
        d = maxim**2/2 + maxim/2
    else:
        d = maxim**2/2
    nC = (d/maxim,d%maxim)
    trans = (nC[0]-center[0],nC[1]-center[1])
    for c in coords:
        pixel = imgArr[c[0]][c[1]]
        newImg[c[0]+trans[0]][c[1]+trans[1]] = pixel
    return (Image.fromarray(numpy.array(newImg,dtype=numpy.uint8),"L"),center[1])

def PasteCharacters(imgArr,labels):
    """
    Takes the imgArray and the labels associated with it and extracts each particular class
        and generates a 100x100 image containing that particular class
    """
    uniqueLabels = []
    for row in labels:
        for label in row:
            if label not in uniqueLabels:
                uniqueLabels.append(label)
    charDict = dict((e1,[]) for e1 in uniqueLabels)
    for i in range(len(labels)):
        row = labels[i]
        for j in range(len(row)):
            label = row[j]
            charDict[label].append((i,j))

    k = charDict.keys()
    imagesWithCenters = []
    for char in k:
        if char != 0:
            imagesWithCenters.append(CreateImage(imgArr,charDict[char]))
    return imagesWithCenters

