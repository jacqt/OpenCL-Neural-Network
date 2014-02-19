#############################################################
# Converts an Image into a 28x28 feature vector             #
# i.e. scales it down to a 28x28 image                      #   
#############################################################

import Image
import ImageEnhance
import numpy
import itertools
import os

def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def normalize(n):
    return (n-177)/100.0# (n - 235)/100.0

def normalize1(n):
    return ((255-n*16)- 177)/100.0

def normalizeVec(image):
    return map (normalize1,list(itertools.chain(*(image))))

def getDist(n1,n2):
    x1 = n1%11
    x2 = n2%11
    y1 = n1/11
    y2 = n2/11
    return (x1-x2)**2 + (y1-y2)**2

def applyTrans(trans,vec):
    newVec = [255]*len(vec) #all whitespace
    g = int(len(vec)**.5)
    for i in range(len(vec)):
        j = i - trans[0] - trans[1]*g
        if j > 0 and j < len(vec):
            newVec[j] = vec[i]

    return newVec

def getCenter(coords):
    "Returns the center of the coords"
    xs = []
    ys = []
    for c in coords:
        xs.append(c[0])
        ys.append(c[1])
    return (sum(xs)/len(xs),sum(ys)/len(ys))

def centerFeature(vec):
    "Centers the feature vector"
    g = int(len(vec)**.5)
    if len(vec)%2 == 0:
        currentCenter = len(vec)/2 + g/2
    else:
        currentCenter = len(vec)/2
        

    xs = []
    ys = []
    for i in range(len(vec)):
        if vec[i] != 255:
            xs.append(i%g)
            ys.append(i/g)

    cx1 = currentCenter%g
    cx2 = sum(xs)/len(xs)
    cy1 = currentCenter/g
    cy2 = sum(ys)/len(ys)
    trans = [cx2-cx1,cy2-cy1]
    return applyTrans(trans,vec)

def getFeatureVec(path):
    "returns the feature vec the image represents"

    image = Image.open(path).convert("L")
    return getFeatureVecFromImg(image)

def getFeatureVecFromImg(image):
    imlist = numpy.asarray(image).tolist()
    imvec = list(itertools.chain(*imlist))
    imvec = centerFeature(imvec)
    imvec = chunks(imvec,int(len(imvec)**.5))

    nlist = numpy.array(imvec, dtype=numpy.uint8)
    #print nlist
    im= Image.fromarray(nlist,"L")

    im2 = im.resize((28,28), Image.ANTIALIAS)
    enhancer = ImageEnhance.Contrast(im2)
    im2 = enhancer.enhance(2.0)

    im2list = numpy.asarray(im2).tolist()
    featureVec = list(itertools.chain(*im2list))

    return featureVec


if __name__ == "__main__":
    curDir = os.getcwd()
    zeroesDir = os.path.join(curDir,"data/test")
    k = getFeatureVec(os.path.join(zeroesDir,"test.png"))
