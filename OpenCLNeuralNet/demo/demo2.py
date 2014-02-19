from Tkinter import *
import neuralnet
import pickle
import convert
import copy
import ExtractCharacters
import Image, ImageDraw
import numpy
import time

import thread
import Queue

points = []
tabspace = "    " #four spaces; i.e. a tab

imageWidth = 400
imageHeight = 300
white = 255
black = 0
CTRLA = "'\\x01'"
CTRLT = "'\\x14'"
netName = "NN-SMALL.net"
lock = thread.allocate_lock()
alreadyWorking = thread.allocate_lock()

image = Image.new("L", (imageWidth,imageHeight), white)
draw = ImageDraw.Draw(image)

def Indicator(n):
    if n > .5: return 1
    else     : return 0

def norm(n):
    return (255-n)/255.0

class Demo(Frame):
    def __init__(self, root, **options):
        #Initialize UI
        Frame.__init__(self,root,**options)
        print "wa"
        optionFrame = Frame(height = 300, width = 200)
        optionFrame.pack(side=LEFT, anchor = N)

        testTypeFrame = Frame(optionFrame, height = 250, width = 200,pady = 5)
        testTypeFrame.pack(side=TOP, anchor=NW)
        testTypeLabel = Label(testTypeFrame, text = "Draw multiple digits for this program to classify!")
        testTypeLabel.pack(side=LEFT, anchor = N)

        buttonFrame = Frame(optionFrame, height = 250, width = 200,pady = 5)
        buttonFrame.pack(side=TOP, anchor=W)
        clear = Button(buttonFrame, text = "Clear - CTRL+A", command = self.clearCanvas)
        clear.config(padx=10, pady=3, anchor=W)
        clear.pack(side=LEFT)
        test = Button(buttonFrame, text = "Test - CTRL+T", command = self.startTestThead)
        test.config(padx=10, pady=3, anchor=W)
        test.pack(side=LEFT)

        resultFrame = Frame(optionFrame, height = 250, width = 200,pady = 5, padx=5)
        resultFrame.pack(side=TOP, anchor=W)
        resultLabel = Label(resultFrame, text = "Results:")
        resultLabel.pack(side=TOP, anchor = W)
        self.resultText = Text(optionFrame,height=12,width=30)
        self.resultText.pack(side=TOP)

        self.c = Canvas(root, bg="white", width = imageWidth, height = imageHeight)
        self.c.configure(cursor="crosshair")
        self.c.pack()

        self.c.bind("<B1-Motion>", self.mouseDownDrag)
        self.c.bind("<ButtonRelease-1>", self.mouseUp)

        self.queue = Queue.Queue()
        self.update_me()

    def update_me(self):
        try:
            while 1:
                outputString = self.queue.get_nowait()
                self.clearText(self.resultText)
                self.resultText.insert(END, outputString)
                self.update_idletasks()
        except Queue.Empty:
            pass
        self.after(100, self.update_me)

    def mouseDownDrag(self,event):
        if len(points) != 0:
            self.c.create_line(points[-1][0], points[-1][1], event.x,event.y, width=6, smooth=1, fill="black")
            draw.line([points[-1], (event.x, event.y)], width=6, fill=black)
        points.append((event.x,event.y))

    def mouseUp(self,event):
        points[:] = []

    def startTestThead(self):
        thread.start_new(Demo.testImage,(self,))

    def clearText(self,textObject):
        textObject.delete(1.0, END)

    def testImage(self):
        print "Testing!"
        self.queue.put("Working...")
        nImage = image.resize((200,150),Image.NEAREST)
        imgArr = numpy.asarray(nImage)
        t = time.time()
        j = numpy.array(ExtractCharacters.LabelPixels(imgArr),dtype=numpy.uint8)
        print time.time()-t
        imagesWithCenters = ExtractCharacters.PasteCharacters(imgArr,j)
        numbers = []
        if alreadyWorking.locked():
            alreadyWorking.release()

        for imageWCenter in imagesWithCenters:
            testVec = map(norm,convert.getFeatureVecFromImg(imageWCenter[0]))
            if alreadyWorking.locked():
                return #another thread has come into play!

            with lock:
                net.ClearNodes()
                net.SetInputs(testVec)
                rawOutputs = net.ComputeOutput()

            outputString =  str(rawOutputs.index(max(rawOutputs)))
            numbers.append((imageWCenter[1],outputString))
        numbers.sort()
        numbers = "".join(map(lambda elem: elem[1], numbers))
        print numbers
        self.queue.put("Neural network read: " + numbers)
        print "done testing"


    def clearCanvas(self):
        self.c.create_rectangle(0,0,imageWidth,imageHeight,fill="white",outline="white")
        draw.rectangle([0,0,imageWidth,imageHeight],fill=white,outline=white)

    def handleKey(self,event):
        print repr(event.char)
        if repr(event.char) == CTRLA:
            self.clearCanvas()
        elif repr(event.char) == CTRLT:
            self.startTestThead()

#Load neural network
t = time.time()
print "Loading neural network..."
net = neuralnet.NeuralNet()
net.LoadData(netName)
print "Done!",
print "Loaded in ", time.time()-t, "seconds.."

root = Tk()
root.title("Digit Recognition")
root.resizable(0,0)

uiFrame = Demo(root)
uiFrame.pack()

root.bind("<Control-Key>",uiFrame.handleKey)
root.mainloop()
