import numpy as np
from numpy import genfromtxt
from math import isclose

numHidNodes = 4 #intial number of hidden nodes will be 2
eta = 0.05 #graidient decent constant

#define the sigmoid function. If the derivative is needed the it returns x*(1-x) which is the derivative of the sigmoid function x=1/(1+np.exp(-x))
def sigmoid(x, derivative=False):
    if(derivative == True):
        return (x*(1-x))

    return 1/(1+np.exp(-x))

def compare(desired_Y,computed_Y): #checks to see if it passed or failed, not working!
    expectedFailed = 0
    expectedPassed = 0
    countFail = 0
    countPass = 0

    for j in range(desired_Y.shape[0]):
        if(desired_Y[j,0] < desired_Y[j,1]):
            expectedPassed += 1
        elif(desired_Y[j,0] > desired_Y[j,1]):
            expectedFailed += 1
    #print("Num of Fail expected: " + str(expectedFailed) + " Num of Passed expected: " + str(expectedPassed))

    for j in range(desired_Y.shape[0]): #iterates through all the rows
        if(computed_Y[j,0] < computed_Y[j,1]):
            countPass += 1
        elif(computed_Y[j,0] > computed_Y[j,1]):
            countFail += 1
    #print("Failed: " + str(countFail) + " Passed: " + str(countPass))

    percentFailed = ((np.abs(expectedFailed-countFail))/(expectedFailed+expectedPassed))*100
    print("Percent incorrect: " + str(percentFailed))

def runNN(inputSet, testSet1, testSet2, testSet3, iterations):
    train_set = genfromtxt(inputSet, delimiter=',')
    X = train_set[0:,0:5] #extracts just input varibles into seperate matrix
    weightMatrixWidth_W = X.shape[1] #number of input variables needed for making wieghts matrix
    X = np.insert(X, 5, 1, axis=1) #appends a 1 to the end of each row so that it can be used to calculate the bias wieght
    Y = train_set[0:,5:]  #extracts just output varibles into seperate matrix
    weightMatrixWidth_V = Y.shape[1]

    np.random.seed(1) #seed random generator
    #creates a random weights matrix between nodes input and hidden. Size is rows= is the length of the number of input varible vector + 1 to account for bias, colums= number of hidden nodes
    W = np.random.random((weightMatrixWidth_W+1,numHidNodes)) #weights from input X to hidden layer
    V = np.random.random((numHidNodes+1,weightMatrixWidth_V)) #weights from hidden layer to output layer

    for i in range(iterations):
        H = sigmoid(np.dot(X,W)) #gives me values for h1 and h2 hiden nodes
        H = np.insert(H, numHidNodes, 1, axis=1) #adds additional 1 for bias variable on output
        y = sigmoid(np.dot(H,V))
        #start back propagation
        error_output = Y - y #computes the diffence in desired versus output from the neural network

        if (i%10000) == 0:
            print("Error: " + str(np.mean(np.abs(error_output))))

        delta_output = error_output * sigmoid(y, derivative=True) #computes the delta at the output layer k

        error_hidden = np.dot(delta_output,V[0:numHidNodes,0:].T) #computes the error or the hidden layer

        delta_hidden = error_hidden * sigmoid(H[0:,0:numHidNodes], derivative=True) #computes the delta of the hidden layer j

        V += eta * np.dot(H.T,delta_output) #update all the weights from the hidden layer to the output layer
        W += eta * np.dot(X.T,delta_hidden)

    compare(Y,y)

    for k in testSet1,testSet2,testSet3:
        test_set = genfromtxt(k, delimiter=',')
        X2 = test_set[0:,0:5] #extracts just input varibles into seperate matrix
        X2 = np.insert(X2, 5, 1, axis=1) #appends a 1 to the end of each row so that it can be used to calculate the bias wieght
        Y2 = test_set[0:,5:]  #extracts just output varibles into seperate matrix

        H2 = sigmoid(np.dot(X2,W)) #gives me values for h1 and h2 hidden nodes
        H2 = np.insert(H2, numHidNodes, 1, axis=1) #adds additional 1 for bias variable on output
        y2 = sigmoid(np.dot(H2,V))
        #print(y2)
        compare(Y2,y2)


#runNN('train1.csv','test1.csv','test2.csv','test3.csv',200000)
#runNN('train2.csv','test1.csv','test2.csv','test3.csv',100000)
runNN('train3.csv','test1.csv','test2.csv','test3.csv',100000)
