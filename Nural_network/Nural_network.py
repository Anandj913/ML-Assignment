'''
Name: Anand Jhunjhunwala
Roll Number: 17EC35032
Coding Assignment 3: Nural Network 

'''
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Configurable parameters
train_data = "Iris.csv"  #Name of train csv file
output_map = {'Iris-setosa': [1.0, 0.0, 0.0], 'Iris-versicolor': [0.0, 1.0, 0.0],  'Iris-virginica': [0.0, 0.0, 1.0]}
train_data_fraction = 0.8
learning_rate = 0.15
num_epoch = 150
np.random.seed(421)
colours = ['r', 'g', 'b', 'y', 'k', 'c']

def get_data(data):
	if os.path.exists(data):
		df = pd.read_csv(data, header=0)
	else:
		print("|-------------------------------------------------------------|")
		print("--The file "+data+" was not found in local folder \n--Please keep the csv file in the same folder as code")
		print("|-------------------------------------------------------------|")
	return df

class NeuralNetwork:
	def __init__(self, x_train, y_train, x_test, y_test, learning_rate, epoch):
		self.input = x_train
		self.target = y_train
		self.input_test = x_test
		self.target_test = y_test
		self.W1 = np.random.rand(4,3)
		self.W2 = np.random.rand(3,3)
		self.b1 = np.random.rand(1,3)
		self.b2 = np.random.rand(1,3)
		self.lr = learning_rate
		self.epoch = epoch
		self.cost_per_epoch = []
		self.train_acc_per_epoch = []
		self.test_acc_per_epoch = []

	@staticmethod
	def Sigmoid(data):
		return 1.0/(1.0 + np.exp(-data))

	@staticmethod
	def derivative_sigmoid(data):
		return data*(1-data)

	def forward_pass(self, input_data):
		self.layer1_out = self.Sigmoid(np.dot(input_data, self.W1) + self.b1)
		output = self.Sigmoid(np.dot(self.layer1_out, self.W2) + self.b2)
		return output

	def back_prop(self, output):
		self.dw2 = np.dot(self.layer1_out.T, ((self.target-output)*self.derivative_sigmoid(output)))
		self.dw1 = np.dot(self.input.T, (np.dot((self.target - output) * self.derivative_sigmoid(output), self.W2.T) * self.derivative_sigmoid(self.layer1_out)))
		self.db2 = np.dot(np.ones((1, self.input.shape[0])), ((self.target-output)*self.derivative_sigmoid(output)))
		self.db1 = np.dot(np.ones((1, self.input.shape[0])), (((self.target - output) * self.derivative_sigmoid(output)*np.sum(self.W2, axis=0))*self.derivative_sigmoid(self.layer1_out)))

		self.W2 += self.dw2*self.lr
		self.W1 += self.dw1*self.lr
		self.b1 += self.db1*self.lr
		self.b2 += self.db2*self.lr

	@staticmethod
	def cost_function(target, output):
		return 0.5*np.sum(np.square(np.subtract(target,output)))

	@staticmethod
	def accuracy(target, output):
		total_corr = 0
		total = len(output)
		for i in range(len(target)):
			if(np.argmax(output[i]) == np.argmax(target[i])):
				total_corr +=1
		return (float(total_corr)/total)*100


	def start_training(self):
		for i in range(self.epoch):
			output = self.forward_pass(self.input)
			self.back_prop(output)
			self.cost_per_epoch.append(self.cost_function(self.target, output))
			self.train_acc_per_epoch.append(self.accuracy(self.target, output))
			self.test_acc_per_epoch.append(self.accuracy(self.target_test, self.forward_pass(self.input_test)))


d = get_data(train_data)
features = list(d.columns[1:5])
output_feature = d.columns[5]
data = d[features]
np_data = np.array(data)
output_data = d[output_feature]

y_data = []
for i in output_data:
	y_data.append(output_map[i])
y_data = np.array(y_data)

normalized_data = (data-data.mean())/data.std()
mean_data = np.array(data.mean())
std_data = np.array(data.std())
x_data = normalized_data.values

data_set = np.hstack((x_data,y_data))
np.random.shuffle(data_set)
total_train_data = int(len(data_set)*train_data_fraction)
total_test_data = len(data_set) - total_train_data
train_set = data_set[:total_train_data]
test_set = data_set[total_train_data:]

x_train = []
y_train = []
x_test = []
y_test = []
for i in range(len(train_set)):
	x_train.append(list(train_set[i][:4]))
	y_train.append(list(train_set[i][4:]))

for i in range(len(test_set)):
	x_test.append(list(test_set[i][:4]))
	y_test.append(list(test_set[i][4:]))

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

nn = NeuralNetwork(x_train, y_train, x_test, y_test, learning_rate, num_epoch)
print("|-------------------------------------------------------------|")
print("|---Starting Training with " + str(num_epoch) + " epoch")
nn.start_training()
print("|---Training Complete")
print("|---Final Training Accuracy: {0}%".format(nn.train_acc_per_epoch[num_epoch-1]))
print("|---Final Test Accuracy: {0}%\n".format(nn.test_acc_per_epoch[num_epoch-1]))


#Classify the example given in problem statement
classify_example = [[4.6, 3.5, 1.8, 0.2], [5.9, 2.5, 1.6, 1.6], [5, 4.2, 3.7, 0.3], [5.7, 4.0, 4.2, 1.2] ]
classify_example = np.array(classify_example)
#normalize the data
classify_example_nor = (classify_example-mean_data)/std_data
classify_output = nn.forward_pass(classify_example_nor)
classify_species = []
for i in range(len(classify_output)):
	if(np.argmax(classify_output[i]) == 0):
		classify_species.append('Iris-setosa')
	elif(np.argmax(classify_output[i]) == 1):
		classify_species.append('Iris-versicolor')
	else:
		classify_species.append('Iris-virginica')

print("|---Classification results of examples")
for i in range(len(classify_output)):
	print("|---Input: {0}".format(classify_example[i]))
	print("|---Output Species: " + classify_species[i] + "\n")

print("|---See plots for rest of the informaion")

ep=[]
for i in range (0,len(nn.cost_per_epoch)):
    ep.append(i)

input_combination = [["sepal_length", "sepal_width"], ["sepal_length", "petal_length"], ["sepal_length","petal_width"], ["sepal_width", "petal_length"], ["sepal_width", "petal_width"], ["petal_length", "petal_width"] ]
un_normalized_data = {"sepal_length": [], "sepal_width": [], "petal_length": [], "petal_width": [] }
norm_data = {"sepal_length": [], "sepal_width": [], "petal_length": [], "petal_width": [] }
for i in range(0, len(np_data)):
	un_normalized_data["sepal_length"].append(np_data[i][0])
	un_normalized_data["sepal_width"].append(np_data[i][1])
	un_normalized_data["petal_length"].append(np_data[i][2])
	un_normalized_data["petal_width"].append(np_data[i][3])
	norm_data["sepal_length"].append(x_data[i][0])
	norm_data["sepal_width"].append(x_data[i][1])
	norm_data["petal_length"].append(x_data[i][2])
	norm_data["petal_width"].append(x_data[i][3])

fig = plt.figure()
fig.suptitle("Unnormalized data", fontweight ="bold", fontsize=30)
for i, c in enumerate(input_combination):
	num = 320 + (i+1)
	k = fig.add_subplot(num)
	k.scatter(un_normalized_data[c[0]], un_normalized_data[c[1]], s =100, c = colours[i])
	k.set_xlabel(c[0], fontweight ="bold", fontsize=15)
	k.set_ylabel(c[1], fontweight ="bold", fontsize=15)
print("|---See plot for unnormalized data")
plt.show()


fig = plt.figure()
fig.suptitle("Normalized data", fontweight ="bold", fontsize=30)
for i, c in enumerate(input_combination):
	num = 320 + (i+1)
	k = fig.add_subplot(num)
	k.scatter(norm_data[c[0]], norm_data[c[1]], s =100, c = colours[i])
	k.set_xlabel(c[0], fontweight ="bold", fontsize=15)
	k.set_ylabel(c[1], fontweight ="bold", fontsize=15)
print("|---See plot for normalized data")
plt.show()


fig, axs = plt.subplots(2)
fig.suptitle('Training results for ' + str(num_epoch) + ' epoch', fontweight ="bold", fontsize=30)
axs[0].plot(ep, nn.cost_per_epoch)
axs[0].set_title('Cost function vs epoch', fontweight ="bold", fontsize=15)
axs[0].set(xlabel = "epoch", ylabel="Cost")
axs[1].plot(ep, nn.train_acc_per_epoch, label="Training Accuracy")
axs[1].plot(ep, nn.test_acc_per_epoch, label="Test Accuracy")
axs[1].set_title('Accuracy vs epoch', fontweight ="bold", fontsize=15)
axs[1].set(xlabel = "epoch", ylabel="Accuracy in %")
axs[1].legend()
print("|---See plot for cost and accuracy information over different epoch")
plt.show()

print("|-------------------------------------------------------------|")