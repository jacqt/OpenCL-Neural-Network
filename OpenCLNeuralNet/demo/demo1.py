from Tkinter import *
import neuralnet
import pickle
import convert
import Image, ImageDraw
import time

import thread # should use the threading module instead!
import Queue

points = []
tabspace = "    " #four spaces; i.e. a tab

imageWidth = 300
imageHeight = 300
white = 255
black = 0
CTRLA = "'\\x01'"
netName = "NN-SMALL.net"

lock = thread.allocate_lock()

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
        testTypeLabel = Label(testTypeFrame, text = "Draw a single digit for this program to classify!")
        testTypeLabel.pack(side=LEFT, anchor = N)

        buttonFrame = Frame(optionFrame, height = 250, width = 200,pady = 5)
        buttonFrame.pack(side=TOP, anchor=W)
        clear = Button(buttonFrame, text = "Clear - CTRL+A", command = self.clearCanvas)
        clear.config(padx=10, pady=3, anchor=W)
        clear.pack(side=LEFT)

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
        self.c.bind("kkk")

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
        thread.start_new(Demo.testDigit,(self,))

    def clearText(self,textObject):
        textObject.delete(1.0, END)

    def testDigit(self):
        print "Testing!"
        try:
            featureVec = map(norm,convert.getFeatureVecFromImg(image))
        except:
            return

        with lock:
            net.ClearNodes()
            net.SetInputs(featureVec)
            rawOutputs = net.ComputeOutput()

        outputs = map (Indicator,rawOutputs)
        roundedOutputs = map (lambda x: "%.3f" % round (x,3),rawOutputs)
        outputString =  "Confidence Values:\n" + \
                tabspace+str(roundedOutputs).ljust(10) + "\n" + \
                "Greatest Confidence:\n"  + \
                tabspace + str(rawOutputs.index(max(rawOutputs))) + " with %3.3f" % (100*max(rawOutputs)) + "% confidence"
        self.queue.put(outputString)
        print "done testing"


    def clearCanvas(self):
        self.c.create_rectangle(0,0,imageWidth,imageHeight,fill="white",outline="white")
        draw.rectangle([0,0,imageWidth,imageHeight],fill=white,outline=white)

    def handleKey(self,event):
        if repr(event.char) == CTRLA:
            self.clearCanvas()

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
