import math
import numpy as np
from PIL import Image
import glob
import os
import cv2
from numba import jit, cuda
from timeit import default_timer as timer
import pygeohash as pgh
import matplotlib.pyplot as plt
import random
#Config
batch_size = 64
num_classes = 205
epochs = 2
im_rows, im_columns, im_depth = 1280, 256, 3
inputLayer = im_rows * im_columns
kernSize = 9
amountOfConvLayers = 2
count = 0




#im = Image.open("C:/Users/jacob/Documents/3DDU2 intelligente systemer/Geoguessr/Data_collection/Data/Train_data_small/6e/img_-23.204195022583008,-65.34710693359375.jpg")

#imageAsArray = np.array(im)/255
#ims = [imageAsArray, imageAsArray, imageAsArray, imageAsArray]


#Layers = [InputLayer,'Conv1', 'Pool1', 'Conv2', 'Pool2',   input_layer_size, int((1280/8)*(256/8)), int((1280/16)*(256/16)), num_classes]
layers = [inputLayer,    4,       4,        4,       4,              72848  , 512, 256, num_classes]

#load weights, biases og kernels eller bare init deres lister, hvis de ikke eksisterer.
weights = []
biases = []
kernels2D = []
tensorKernels = []
wPath = 'C:/Users/jacob/Documents/3DDU2 intelligente systemer/Geoguessr/weights.npy'
bPath = 'C:/Users/jacob/Documents/3DDU2 intelligente systemer/Geoguessr/biases.npy'
k2DPath = 'C:/Users/jacob/Documents/3DDU2 intelligente systemer/Geoguessr/kernels2D.npy'
tKPath = 'C:/Users/jacob/Documents/3DDU2 intelligente systemer/Geoguessr/tensorKernels.npy'

if os.path.exists(wPath):
    weights = np.load(wPath, allow_pickle=True)
    
if os.path.exists(bPath):
    biases = np.load(bPath, allow_pickle=True)
    
if os.path.exists(k2DPath):
    kernels2D = np.load(k2DPath, allow_pickle=True)

if os.path.exists(tKPath):
    tensorKernels = np.load(tKPath, allow_pickle=True)
    
#Preprocess
def preprocess(imageAdress):
    curImg = Image.open(imageAdress)
    imPreprocessed = np.array(curImg)/255
    return imPreprocessed

#Find billeder
dirpath = "C:/Users/jacob/Documents/3DDU2 intelligente systemer/Geoguessr/Data_collection/Data/Train_data_small" #"C:/Users/jacob/Documents/3DDU2 intelligente systemer/Geoguessr/Data_collection/Data/testing_data_for_test"
Train_data = glob.glob(dirpath + "/*/*.jpg")
labels = glob.glob(dirpath + "/*")
labels = [label[len(dirpath):][:3] for label in labels]
Train_data_labels = [x[len(dirpath):][:3] for x in Train_data]
for index, label in enumerate(Train_data_labels):
    Train_data_labels[index] = labels.index(label)

Images = []

for i in range(10): #int(len(Train_data)/32) #len(Train_data)
    singleImage = [np.array(Image.open(Train_data[i]))/255]
    y = np.zeros(np.array(labels).shape)
    y[Train_data_labels[i]] = 1
    singleImage.append(y)
    Images.append(singleImage)

random.Random(4).shuffle(Images)
    
#print('færdig med at loade: ', len(Images), ' billeder')

batchCount = len(Train_data)/batch_size
if type(batchCount) == float:
    batchCount = int(batchCount)+1

#opdel i batches
for batch in range(0, batchCount):
    curBatch = Train_data[(batch*batch_size):(batch*batch_size)+batch_size]

#2/3D Conv (Convolute 3D tensor af im_rows, im_columns, im_depth --> 2D matrix som kan laves til vector column for hurtigt calc)
@jit(target_backend='GPU')
def conv2D(imageAsArray, kernel, biases=0, stride=1, padding=0):
    #Cross correlation på kernel
    kernel = np.flipud(np.fliplr(kernel))
    
    #Shapes af kernel og billede
    kernelWidth = kernel.shape[0]
    kernelHeight = kernel.shape[1]
    kernelDepth = kernel.shape[2]
    imageWidth = imageAsArray.shape[0]
    imageHeight = imageAsArray.shape[1]
    imageDepth = imageAsArray.shape[2]
    
    #Shape af output convolutionen
    xOutput = int(((imageWidth - kernelWidth + 2*padding) / stride)+1)
    yOutput = int(((imageHeight - kernelHeight + 2 * padding) / stride)+1)
    zOutput = int(((imageDepth - kernelDepth + 2 * padding) / stride)+1)
    output = np.zeros((xOutput, yOutput, zOutput))
    
    #Tilføj zero-padding, hvis man vil have det (WIP)
    if padding != 0:
        imagePadded = np.zeros((imageWidth + padding*2, imageHeight + padding*2, imageDepth))
        imagePadded[int(padding):int(-1*padding), int(padding):int(-1*padding), 0:3] = imageAsArray
        print(imagePadded)
    else:
        imagePadded = imageAsArray
        
    #iterate igennem billedet med kernel
    for z in range(imageDepth):
        for y in range(imageHeight):
            #print(y)
            if y > imageHeight - kernelHeight:
                break
            if y % stride == 0:
                for x in range(0, imageWidth):
                    #print(y)
                    if x > imageWidth - kernelWidth:
                        break
                    try:
                        if x % stride == 0:
                        #print(x)
                            output[x, y, z] = ((kernel * imagePadded[x: x + kernelWidth, y: y + kernelHeight, z:z+kernelDepth]).sum()) / (kernelWidth * kernelHeight * kernelDepth)
                    except:
                        break
    return output

#Pool2D
def pool2D(imAsArray):
    xLen, yLen = imAsArray.shape[0], imAsArray.shape[1]
    poolX = 2
    poolY = 2
    
    xLenFloored = xLen // poolX
    yLenFloored = yLen // poolY
    
    pooled = imAsArray[:xLenFloored*poolX, :yLenFloored*poolY].reshape(xLenFloored, poolX, yLenFloored, poolY, 1).max(axis=(1, 3))
    return pooled


#Activation function (sigmoid)
@jit(target_backend='cuda')
def sigmoid(x):
    #return np.where(x > 0, x, x * 0.0001)
    return 1/(1+np.exp(-x))

@jit(target_backend='cuda')
def sigmoidprime(x):
    #return np.where(x > 0, 1, 0.0001)    
    return sigmoid(x)*(1-sigmoid(x))

#Save W/b
def saveVars():
    b = np.array(biases, dtype=object)
    np.save(bPath, b)
    w = np.array(weights, dtype=object)
    np.save(wPath, w)
    return w, b

#Gem kernels
def saveKernels():
    tK = np.array(tensorKernels)
    np.save(tKPath, tK)
    k2D = np.array(kernels2D)
    np.save(k2DPath, k2D)
    return tK, k2D

#Init W/b
#biases ligger i array med shape (10,) og er indexet som hvist her: [layer 0-9] hvor det er eksklusiv inputlayer, [neuron 0-layer[layer#]]
#Weights ligger i array indexet på følgende måde [layer 0-2] hvor det er eksklusiv første (Tænk connections baglæns) [neuron 0 - layer[layer#]], [connection 0 - layer[layer#-1]]
def InitWB():
    for i in layers[1:]:
        np.random.seed(1)
        biases.append(np.random.randn(i, 1))
        
    for i, j  in zip(layers[-4:-1], layers[-3:]):
        np.random.seed(1)
        weights.append(np.random.randn(j, i))
        
    wI, bI = saveVars()
    
    return wI, bI

if len(weights) < 2:
    weights, biases = InitWB()


#Init kernels
#De ligger i array i formen [layer 0-1] [filter 0-15] [x 0-8] [y 0-8] [z 0-2]
def InitKernels():
    for i in [layers[3]]:
        np.random.seed(2)
        kernels2D.append([np.random.normal(loc=1, scale=6, size=(kernSize, kernSize, 1)) for j in range(0, i)])
    for q in [layers[1]]:
        np.random.seed(1)
        tensorKernels.append([np.random.normal(loc=1, scale=6, size=(kernSize, kernSize, 3)) for l in range(0, q)])
        
        
if len(kernels2D) < 1:
    InitKernels()
    tensorKernels, kernels2D = saveKernels()

#Mean Cost
def cost(activations, y):
    #print(activations.shape)
    #print('a', activations.shape, 'y', y.shape)
    #print((np.subtract(activations, y.reshape(205,1))**2).shape)
    return np.subtract(activations, y.reshape(205,1))**2

#første afledte af cost funktionen
def cost_derivative(activations, y):
    print(y[0:4].flatten())
    #print((2*np.subtract(activations,y)).shape)
    return 2*np.subtract(activations,y)

#backprop
def backprop(imAsArray, label):
    collectionOfSplit = []
    splitMerge = []
    for index, amountInConvLayer in enumerate(layers[1:amountOfConvLayers*2:2]):
        if index == 0:
            for filter in range(0, amountInConvLayer):
                collectionOfSplit.append(conv2D(imAsArray, tensorKernels[0][filter]))
                collectionOfSplit[filter] = pool2D(collectionOfSplit[filter])
                
        else:
            for filter in range(0, amountInConvLayer):
                collectionOfSplit[filter] = conv2D(collectionOfSplit[filter], kernels2D[0][filter])
                collectionOfSplit[filter] = pool2D(collectionOfSplit[filter])
                cv2.imwrite('Convoluted%s.jpg' % filter , np.array(collectionOfSplit[filter])*255)
                #collectionOfSplit[filter] = conv2D(collectionOfSplit[filter], kernels2D[0][filter])
                
    splitMerge = np.array(collectionOfSplit).flatten()
    a = splitMerge
    activation = a.reshape(72848, 1)
    activations = [a.reshape(72848, 1)]
    zList = []
    gradient_b = [np.zeros(b.shape) for b in biases]
    gradient_w = [np.zeros(w.shape) for w in weights]
    
    for b, w in zip(biases[-3:], weights):
        #print('w.shape', w.shape, 'b.shape', b.shape, 'a.shape', activation.shape)
        z = np.dot(w, activation)+b
        activation = sigmoid(z)
        #activation = activation.reshape(len(activation), 1)
        #activation = activation.reshape(activation.shape, 1)
        zList.append(z)
        activations.append(activation.reshape(len(activation), 1))
    

    #print(activations[-1].shape)
    delta = cost_derivative(activations[-1], label.reshape(205, 1)) * sigmoidprime(zList[-1].reshape(205,1))
    #print(delta.shape)
    gradient_b[-1] = delta
    #print(activations[-2])
    #print(activations[-2].shape)
    gradient_w[-1] = np.dot(delta, activations[-2].transpose())
    
    for layer in range(2, len(layers[-4:])):
        z = zList[-layer].reshape(len(zList[-layer]), 1)
        sp = sigmoidprime(z)
        delta = np.dot(weights[-layer+1].transpose(), delta) * sp
        gradient_b[-layer] = delta
        gradient_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
    #print('b')
    #for bia in gradient_b:
    #    print(bia.shape)
    #print('w')
    #for wei in gradient_w:
    #    print(wei.shape)
    return (gradient_b, gradient_w)

#Feedforward
def feedForward(imAsArray):
    collectionOfSplit = []
    splitMerge = []
    for index, amountInConvLayer in enumerate(layers[1:amountOfConvLayers*2:2]):
        if index == 0:
            for filter in range(0, amountInConvLayer):
                #keras.layer.Conv2D(32, (9, 9), activation='relu', inputshape=(256,1280,3))(ims)
                collectionOfSplit.append(conv2D(imAsArray, tensorKernels[0][filter]))
                collectionOfSplit[filter] = pool2D(collectionOfSplit[filter])
        else:
            for filter in range(0, amountInConvLayer):
                collectionOfSplit[filter] = conv2D(collectionOfSplit[filter], kernels2D[0][filter])
                collectionOfSplit[filter] = pool2D(collectionOfSplit[filter])
                #collectionOfSplit[filter] = conv2D(collectionOfSplit[filter], kernels2D[0][filter])
                
                
    splitMerge = np.array(collectionOfSplit).flatten()
    a = splitMerge.reshape(72848,1)
    #start = timer()
    for b, w in zip(biases[-3:], weights):
        #print(w.shape, b.shape, a.shape)
        a = sigmoid(np.add(np.dot(w, a),b))
    #print("With GPU: ", timer()-start)
    return a

#print(tensorKernels[2].shape)
#print(singleImage[1])
epochs = 200
eta = 0.01
costList = []
#print(feedForward(Images[0][0]))
#print(np.argsort(feedForward(Images[0][0]).flatten()))
costList.append((cost(feedForward(Images[0][0]), Images[0][1]).sum()+cost(feedForward(Images[1][0]), Images[1][1]).sum()+cost(feedForward(Images[2][0]), Images[2][1]).sum())/3)
print(costList[0])
#print('weight shapes')
#for thing in weights:
#    print(thing.shape)
for n in range(epochs):
    start = timer()
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    for i in range(0, 3):#len(Images)):
        #q = backprop(Images[i][0], Images[i][1])
        delta_gradient_b, delta_gradient_w = backprop(Images[i][0], Images[i][1])
        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_gradient_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_gradient_w)]
        #print('der er noget galt med shape af gradient og sådan')
        #print([e.shape for e in biases])
        #print([å.shape for å in weights])
        #weights = [np.subtract(w, (eta * nw)) for w, nw in zip(weights, delta_gradient_w)]
        #biases =  [np.subtract(b, (eta * nb)) for b, nb in zip(biases, delta_gradient_b)]
        #print('efter gradient change')
        #print([e.shape for e in biases])
        #print([å.shape for å in weights])
    weights = [np.subtract(w, ((eta/3) * (n+31) * nw)) for w, nw in zip(weights, nabla_w)]
    biases =  [np.subtract(b, ((eta/3) * (n+31) * nb)) for b, nb in zip(biases, nabla_b)]
    saveVars()
    print("With GPU: ", timer()-start)
    #print(feedForward(Images[0][0]))
    costList.append((cost(feedForward(Images[0][0]), Images[0][1]).sum()+cost(feedForward(Images[1][0]), Images[1][1]).sum()+cost(feedForward(Images[2][0]), Images[2][1]).sum())/3)
    print(costList[n+1])
    #print('biases: ', biases[-1][0:4])

#print(Images[0][1])
#print(labels[0][-2:])
for integer in range(3):#len(Images)):
    pap = feedForward(Images[integer][0])
    #print(np.array(pap))
    print('billede', integer,'har labellen:', labels[np.argsort(np.array(Images[integer][1][0:3]))[-1]][-2:])
    for o in range(1,6):
        #print(pap[np.argsort(np.array(pap))[-o]])
        print('gæt #', o, 'er: ', labels[np.argsort(pap.flatten())[-o]][-2:], ' for billede #', integer)
    #print(pgh.decode(labels[np.argsort(np.array(pap))[-1]][-2:]))
xList = np.arange(len(costList))
plot = plt.figure()
plot.set_figheight(5)
plot.set_figwidth(5)
plt.plot(xList, costList)
plt.show()
#results = np.array(feedForward(imageAsArray))
#result = np.argsort(results)[0]
#print(pgh.decode(labels[result][-2:]))
#print(collectionOfSplit.shape)
#cv2.imwrite('conv2D.jpg', collectionOfSplit[0]*255)
