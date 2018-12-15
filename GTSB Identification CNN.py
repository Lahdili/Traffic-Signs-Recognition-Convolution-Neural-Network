#########################################################
#            Traffic signs recognition                  #
#########################################################

# import the necessary packages

import os
import time
import cPickle
from sklearn.utils import shuffle

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import itertools

from lasagne import random as lasagne_random
from lasagne import layers
from lasagne.nonlinearities import softmax, tanh
from lasagne import objectives
from lasagne import updates
import lasagne

import theano
import theano.tensor as T

import warnings
warnings.filterwarnings("ignore")


######################## CONFIG #########################

#Fixed random seed
RANDOM_SEED = 1337
RANDOM = np.random.RandomState(RANDOM_SEED)
lasagne_random.set_rng(RANDOM)

#Training settings
EPOCHS = 20
LR_START = 0.0005
LR_END = 0.000001

################### DATASET HANDLING ####################
DATASET_PATH = 'GSTB_Dataset'
def parseDataset():

    #Subfolders are used as class labels
    classes = [folder for folder in sorted(os.listdir(DATASET_PATH))]

    #Enlisting all image paths
    images = []
    for c in classes:
        images += ([os.path.join(DATASET_PATH, c, path) for path in os.listdir(os.path.join(DATASET_PATH, c))])

    #Shuffling image paths
    images = shuffle(images, random_state=42)

    #15% of the datasets is used as validation data
    vsplit = int(len(images) * 0.15)
    train = images[:-vsplit]
    val = images[-vsplit:]

    #Print stats
    print "CLASS LABELS:", classes
    print "TRAINING IMAGES:", len(train)
    print "VALIDATION IMAGES:", len(val)

    return classes, train, val

#Parse dataset
CLASSES, TRAIN, VAL = parseDataset()

#################### BATCH HANDLING #####################
def loadImageAndTarget(path):

    #resize image into 32x32 pixels
    try:
        img = cv2.imread(path)
        img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)
    except ValueError:
        print("Error in image ", path)
    
    #The images has the shape (32, 32, 3) but we need it to be (3, 32, 32)
    img = np.transpose(img, (2, 0, 1))
    
    #Subfolders are used as class labels
    label = path.split("/")[-2]

    #The index of the label is extracted from the CLASSES
    index = CLASSES.index(label)

    #allocate array for target, we have 43 classes
    target = np.zeros((43), dtype='float32')

    #The target array is setted to = 1.0 at the index of the label , all other entries remain zero
    #Example: if label = "Sign X" and "Sign X" has index 1 in CLASSES, target looks like: [0.0, 1.0, 0.0, 0.0, 0.0, ...]
    target[index] = 1.0

    # A 4D-vector for the image and a 2D-vector for targets
    img = img.reshape(-1, 3, 32, 32)
    target = target.reshape(-1, 43)

    return img, target

#The batch size is fixed to 128
BATCH_SIZE = 128

def getDatasetChunk(split):

    # Dataset is divided into batch-sized chunks
    for i in xrange(0, len(split), BATCH_SIZE):
        yield split[i:i+BATCH_SIZE]

def getNextImageBatch(split=TRAIN):    

    # Allocate numpy arrays for image data and targets
    # Input shape of the CNN is (None, 3, 32, 32)
    x_b = np.zeros((BATCH_SIZE, 3, 32, 32), dtype='float32')
    # Output shape ofthe CNN is (None, 43) as we have 43 classes
    y_b = np.zeros((BATCH_SIZE, 43), dtype='float32')

    #fill batch
    for chunk in getDatasetChunk(split):        
        count = 0
        for path in chunk:
            #load image data and class label from path
            x, y = loadImageAndTarget(path)

            #pack into batch array
            x_b[count] = x
            y_b[count] = y
            count += 1


        yield x_b[:len(chunk)], y_b[:len(chunk)]

################## BUILDING THE MODEL ###################
def buildModel():

    # Theinput layer with the inputs (None, dimensions, width, height)
    l_input = layers.InputLayer((None, 3, 32, 32))

    # First convolutional layer, has l_input layer as incoming and is followed by a pooling layer, filters = 16
    l_conv1 = layers.Conv2DLayer(l_input, num_filters=16, filter_size=3, pad='same', nonlinearity=tanh)
    l_pool1 = layers.MaxPool2DLayer(l_conv1, pool_size=2)

    # The second convolution (l_pool1 is incoming), filters = 32
    l_conv2 = layers.Conv2DLayer(l_pool1, num_filters=32, filter_size=3, pad='same', nonlinearity=tanh)
    l_pool2 = layers.MaxPool2DLayer(l_conv2, pool_size=2)

    # The third convolution (l_pool2 is incoming), filters = 64
    l_conv3 = layers.Conv2DLayer(l_pool2, num_filters=64, filter_size=3, pad='same', nonlinearity=tanh)
    l_pool3 = layers.MaxPool2DLayer(l_conv3, pool_size=2)

    # The fourth and final convolution, filters = 128
    l_conv4 = layers.Conv2DLayer(l_pool3, num_filters=128, filter_size=3, pad='same', nonlinearity=tanh)
    l_pool4 = layers.MaxPool2DLayer(l_conv4, pool_size=2)

    # The CNN contains 3 dense layers, one of them is the output layer
    l_dense1 = layers.DenseLayer(l_pool4, num_units=64, nonlinearity=tanh)
    l_dense2 = layers.DenseLayer(l_dense1, num_units=64, nonlinearity=tanh)

    # The output layer has 43 units which is exactly the count of our class labels
    # It has a softmax activation function, its values represent class probabilities
    l_output = layers.DenseLayer(l_dense2, num_units=43, nonlinearity=softmax)

    #print "The CNN model has ", layers.count_params(l_output), "parameters"

    # Returning the layer stack by returning the last layer
    return l_output

NET = buildModel()


#################### LOSS FUNCTION ######################
def calc_loss(prediction, targets):

    # Categorical crossentropy is the best choice for a multi-class softmax output
    l = T.mean(objectives.categorical_crossentropy(prediction, targets))
    
    return l

# Theano variable for the class targets
# The output vector the CNN should predict
targets = T.matrix('targets', dtype=theano.config.floatX)

# Compute the network output
prediction = layers.get_output(NET)

# Calculate the loss
loss = calc_loss(prediction, targets)

################# ACCURACY FUNCTION #####################
def calc_accuracy(prediction, targets):

    # The lasagne objective categorical_accuracy is used to determine the top1 accuracy
    a = T.mean(objectives.categorical_accuracy(prediction, targets, top_k=1))
    
    return a

accuracy = calc_accuracy(prediction, targets)

####################### UPDATES #########################
# Get all trainable parameters (weights) of the CNN
params = layers.get_all_params(NET, trainable=True)

# The dynamic learning rate is applied during the training process
lr_dynamic = T.scalar(name='learning_rate')

# The adam update methode is used to update the  params based on the loss function & the learning rate
param_updates = updates.adam(loss, params, learning_rate=lr_dynamic)

#################### TRAIN FUNCTION ######################
# The theano train functions takes images and class targets as input
# It updates the parameters of the net and returns the current loss as float value

# Compiling theano functions 
#print "COMPILING THEANO TRAIN FUNCTION...",
train_net = theano.function([layers.get_all_layers(NET)[0].input_var, targets, lr_dynamic], loss, updates=param_updates)


################# PREDICTION FUNCTION ####################
# The prediction function is used to calculate the validation accuracy
# First the CNN's output is retrieved
net_output = layers.get_output(NET)

# Compiling theano test function
print "COMPILING THEANO TEST FUNCTION...",
test_net = theano.function([layers.get_all_layers(NET)[0].input_var, targets], [net_output, loss, accuracy])

##################### STAT PLOT #########################
plt.ion()
def showChart(epoch):

    #new figure
    plt.figure(0)
    plt.clf()

    #x-Axis = epoch
    e = range(0, epoch)

    #loss subplot
    plt.subplot(211)
    #plt.plot(e, train_loss, 'r-', label='Train Loss')
    plt.plot(e, val_loss, 'b-', label='Val Loss')
    plt.ylabel('loss')
    
    
    #show labels
    plt.legend(loc='upper right', shadow=True)

    #accuracy subplot
    plt.subplot(212)
    plt.plot(e, val_accuracy, 'g-')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    #show
    plt.show()
    plt.pause(0.5)





###################### TRAINING #########################
print "START TRAINING..."

train_loss = []
val_loss = []
val_accuracy = []


for epoch in range(1, EPOCHS + 1):

    # Start timer
    start = time.time()


    # Determine current dynamic learning rate
    learning_rate = LR_START - (epoch - 1) * ((LR_START - LR_END) / (EPOCHS - 1))

    # Iterate over train split batches and calculate mean loss for epoch
    t_l = []
    for image_batch, target_batch in getNextImageBatch():

        # Calling the training functions returns the current loss
        l = train_net(image_batch, target_batch, learning_rate)
        t_l.append(l)
        
    
    
    # Validate the CNN every epoch and pass the validation split through as well
    v_l = []
    v_a = []
    for image_batch, target_batch in getNextImageBatch(VAL):

        # Calling the test function returns the net output, loss and accuracy
        prediction_batch, l, a = test_net(image_batch, target_batch)
        v_l.append(l)
        v_a.append(a)

    # Stop timer
    end = time.time()

    # Calculate stats for epoch
    train_loss.append(np.mean(t_l))
    val_loss.append(np.mean(v_l))
    val_accuracy.append(np.mean(v_a))

    # Print stats for epoch
    print "EPOCH:", epoch,
    print "TRAIN LOSS:", train_loss[-1],
    print "VAL LOSS:", val_loss[-1],
    print "VAL ACCURACY:", (int(val_accuracy[-1] * 1000) / 10.0), "%",
    print "TIME:", (int((end - start) * 10) / 10.0), "s\n"
    print "LEARNING RATE:", learning_rate

    # Show chart
    showChart(epoch)


print "TRAINING DONE!"

# Show best accuracy and epoch
print "BEST VAL ACCURACY:", (int(max(val_accuracy) * 1000) / 10.0), "%", "EPOCH:", val_accuracy.index(max(val_accuracy)) + 1



