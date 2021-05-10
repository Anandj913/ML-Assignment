#Name: Shivam Saxena
#Roll no.: 17EC10054
#Assignment : Neural Network


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

accu_train = []
accu_test = []
cost = []

def NeuralNetwork(X_train, Y_train, X_test, Y_test, X_val=None, Y_val=None, epochs=10, nodes=[], lr=0.15):
    hidden_layers = len(nodes) - 1
    weights = InitializeWeights(nodes)

    for epoch in range(1, epochs+1):
        weights = Train(X_train, Y_train, lr, weights)

        accu_train.append(Accuracy(X_train, Y_train, weights))
        accu_test.append(Accuracy(X_test, Y_test, weights))
        if(epoch % 20 == 0):
            print("Epoch {}".format(epoch))
            print("Training Accuracy:{}".format(Accuracy(X_train, Y_train, weights)))
            # if X_val.any():
            #     print("Validation Accuracy:{}".format(Accuracy(X_val, Y_val, weights)))
            
    return weights

def InitializeWeights(nodes):
    """Initialize weights with random values in [-1, 1] (including bias)"""
    layers, weights = len(nodes), []
    
    for i in range(1, layers):
        w = [[np.random.uniform(-1, 1) for k in range(nodes[i-1] + 1)]
              for j in range(nodes[i])]
        weights.append(np.matrix(w))
    
    return weights

def ForwardPropagation(x, weights, layers):
    activations, layer_input = [x], x
    for j in range(layers):
        activation = Sigmoid(np.dot(layer_input, weights[j].T))
        activations.append(activation)
        layer_input = np.append(1, activation) # Augment with bias
    
    return activations

def BackPropagation(y, activations, weights, layers):
    outputFinal = activations[-1]
    error = np.matrix(y - outputFinal) # Error at output
    cerror = np.square(error)
    cerror = 0.5*np.sum(cerror)
    for j in range(layers, 0, -1):
        currActivation = activations[j]
        
        if(j > 1):
            # Augment previous activation
            prevActivation = np.append(1, activations[j-1])
        else:
            # First hidden layer, prevActivation is input (without bias)
            prevActivation = activations[0]
        
        delta = np.multiply(error, SigmoidDerivative(currActivation))
        weights[j-1] += lr * np.multiply(delta.T, prevActivation)

        w = np.delete(weights[j-1], [0], axis=1) # Remove bias from weights
        error = np.dot(delta, w) # Calculate error for current layer
    
    return weights, cerror

def Train(X, Y, lr, weights):
    layers = len(weights)
    totalcost = 0;
    for i in range(len(X)):
        x, y = X[i], Y[i]
        x = np.matrix(np.append(1, x)) # Augment feature vector
        
        activations = ForwardPropagation(x, weights, layers)
        weights, error = BackPropagation(y, activations, weights, layers)
        totalcost += error
    cost.append(totalcost)
    return weights

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def SigmoidDerivative(x):
    return np.multiply(x, 1-x)

def Predict(item, weights):
    layers = len(weights)
    item = np.append(1, item) # Augment feature vector
    
    ##_Forward Propagation_##
    activations = ForwardPropagation(item, weights, layers)
    
    outputFinal = activations[-1].A1
    index = FindMaxActivation(outputFinal)

    # Initialize prediction vector to zeros
    y = [0 for i in range(len(outputFinal))]
    y[index] = 1  # Set guessed class to 1

    return y # Return prediction vector


def FindMaxActivation(output):
    """Find max activation in output"""
    m, index = output[0], 0
    for i in range(1, len(output)):
        if(output[i] > m):
            m, index = output[i], i
    
    return index

def Accuracy(X, Y, weights):
    """Run set through network, find overall accuracy"""
    correct = 0

    for i in range(len(X)):
        x, y = X[i], list(Y[i])
        guess = Predict(x, weights)

        if(y == guess):
            # Guessed correctly
            correct += 1

    return (float(correct)/ len(X))*100



dataset=pd.read_csv("Iris.csv")

X = dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
X = np.array(X)
X_cols=np.shape(X)[1]
X_rows=np.shape(X)[0]

one_hot_encoder = OneHotEncoder(sparse=False)

Y = dataset.Species
Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))


means=np.zeros(X_cols)
devs=np.zeros(X_cols)

for i in range(X_cols):
    means[i]=np.mean(X[:,i])
    #print(np.mean(X[:,i]))
    #print(np.std(X[:,i]))

    devs[i]=np.std(X[:,i])
    for j in range(X_rows):
        X[j,i]=(X[j,i]-means[i])/devs[i]
    #X[0:2,i]=(X[0:2,i]-np.mean(X[0:2,i]))/np.std(X[0:2,i])

# print(X[0:3,:])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1)

f = len(X[0]) # Number of features
o = len(Y[0]) # Number of outputs / classes

layers = [f, 4, o] # Number of nodes in layers
lr, epochs = 0.15, 100

weights = NeuralNetwork(X_train, Y_train, X_test, Y_test, epochs=epochs, nodes=layers, lr=lr);

print("Testing Accuracy: {}".format(Accuracy(X_test, Y_test, weights)))

classify=np.array([[4.6,3.5,1.8,0.2],
                [5.9,2.5,1.6,1.6],
                [5,4.2,3.7,0.3],
                [5.7,4,4.2,1.2]])


classify=np.array(classify)

c_rows=np.shape(classify)[0]
c_cols=np.shape(classify)[1]
for i in range(c_rows):
    for j in range(c_cols):
        classify[j,i]=(classify[j,i]-means[i])/devs[i]

#arr = [[0]*cols]*rows
cl_out=[[0]*o]*c_rows


cl_out=np.array(cl_out)

print("Classification results: ")
for i in range(c_rows):
    cl_out[i,:]=Predict(classify[i,:],weights)
    if((cl_out[i,:]==np.array([1,0,0])).all()):
        print("Input: {} Species: Iris-setosa".format(classify[i,:]))
    elif((cl_out[i,:]==np.array([0,1,0])).all()):
        print("Input: {} Species: Iris-versicolor".format(classify[i,:]))
    else:
        print("Input: {} Species: Iris-virginica".format(classify[i,:]))

print("See Graphs")
ep = []
for i in range(1, len(cost)+1):
    ep.append(i)

fig, a = plt.subplots(2)
a[0].plot(ep, cost)
a[0].set_title('Cost function vs epoch')
a[0].set(xlabel = "epoch", ylabel="Cost")
a[1].plot(ep, accu_train, label="Training Accuracy")
a[1].plot(ep, accu_test, label="Test Accuracy")
a[1].set_title('Accuracy vs epoch')
a[1].set(xlabel = "epoch", ylabel="Accuracy in %")
a[1].legend()
plt.show()