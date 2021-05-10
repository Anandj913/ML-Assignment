'''
Name: Anand Jhunjhunwala
Roll Number: 17EC35032
Coding Assignment 4: K-NN
'''
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import math
try:
   from queue import PriorityQueue
except ImportError:
   from Queue import PriorityQueue


# Configurable parameters
train_data = "cancer_dataset.csv"  #Name of train csv file
train_data_fraction = 0.85
K_for_NN = [1,3,5,7]
np.random.seed(300)

def get_data(data):
	if os.path.exists(data):
		df = pd.read_csv(data)
	else:
		print("|-------------------------------------------------------------|")
		print("--The file "+data+" was not found in local folder \n--Please keep the csv file in the same folder as code")
		print("|-------------------------------------------------------------|")
	return df

def Euclidean_distance(p,q):
	d = np.sum(np.square(np.subtract(p,q)))
	return math.sqrt(d)

def Cosine_similarity(p,q):
	num = np.dot(p,q)
	deno = math.sqrt(np.sum(np.square(p)))*math.sqrt(np.sum(np.square(q)))
	#1-cosineilarity is returned because higher the cosine similarity more similar the data
	return 1 - float(num)/deno

def Euclidean_distance_norm(p, q, mean_data, std_data):
	p = (p-mean_data)/std_data
	q = (q-mean_data)/std_data
	return Euclidean_distance(p, q)

def prediction(q, k):
	m = {2:0, 4:0}
	for i in range(k):
		data = q.get()
		m[data[-1]] = m[data[-1]] + 1
	if m[2]>=m[4]:
		return 2
	else:
		return 4
def calculate_accuracy(pred_class, true_class):
	total_correct = 0
	for i in range(len(true_class)):
		if pred_class[i]==true_class[i]:
			total_correct = total_correct + 1
	total_data = len(true_class)
	return (float(total_correct)/total_data)*100

d = get_data(train_data)
features = list(d.columns[1:11])
features_norm = list(d.columns[1:10])

data_set = d[features]
data_set_for_norm = d[features_norm]

mean_data = np.array(data_set_for_norm.mean())
std_data =  np.array(data_set_for_norm.std())

np_data_set = data_set.values
np.random.shuffle(np_data_set)

total_train_data = int(len(np_data_set)*train_data_fraction)
train_set = np_data_set[:total_train_data]
test_set = np_data_set[total_train_data:]


accuracy_for_different_K = {}
print("|-------------------------------------------------------------|")
print("|--Number of Traning Data: {0}".format(total_train_data))
print("|--Number of Testing Data: {0}".format(len(np_data_set) - total_train_data))
for K in K_for_NN:
	print("|--Processing for K = {0}".format(K))
	pred_accuracy = []
	pred_class_euclidean = []
	pred_class_cosine = []
	pred_class_euclidean_norm = []
	true_class = []
	for t in test_set:
		q_euclidean = PriorityQueue()
		q_cosine = PriorityQueue()
		q_euclidean_norm = PriorityQueue()
		for k in train_set:
			q_euclidean.put([Euclidean_distance(t[:-1], k[:-1]), k[-1]])
			q_cosine.put([Cosine_similarity(t[:-1], k[:-1]), k[-1]])
			q_euclidean_norm.put([Euclidean_distance_norm(t[:-1], k[:-1], mean_data, std_data), k[-1]])
		pred_class_euclidean.append(prediction(q_euclidean, K))
		pred_class_cosine.append(prediction(q_cosine, K))
		pred_class_euclidean_norm.append(prediction(q_euclidean_norm, K))
		true_class.append(t[-1])
	pred_accuracy.append(calculate_accuracy(pred_class_euclidean, true_class))
	pred_accuracy.append(calculate_accuracy(pred_class_euclidean_norm, true_class))
	pred_accuracy.append(calculate_accuracy(pred_class_cosine, true_class))
	accuracy_for_different_K[K] = pred_accuracy[:]
print("|--Accuracy Results:")
for a in accuracy_for_different_K.keys():
	print("\n|--For K = {0} and Distance matrix".format(a))
	print("|     Euclidean: {:.2f}%".format(accuracy_for_different_K[a][0]))
	print("|     Normalized Euclidean: {:.2f}%".format(accuracy_for_different_K[a][1]))
	print("|     Cosine Similarity: {:.2f}%".format(accuracy_for_different_K[a][2]))

euclidean_accuracy = []
norm_euclidean_accuracy = []
cosine_accuracy = []
for a in accuracy_for_different_K.keys():
	euclidean_accuracy.append(accuracy_for_different_K[a][0])
	norm_euclidean_accuracy.append(accuracy_for_different_K[a][1])
	cosine_accuracy.append(accuracy_for_different_K[a][2])
# set width of bar 
barWidth = 0.15
fig = plt.subplots(figsize =(10, 6)) 
   
   
# Set position of bar on X axis 
br1 = np.arange(len(accuracy_for_different_K.keys()))
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
   
# Make the plot 
plt.bar(br1, euclidean_accuracy, color ='r', width = barWidth, 
        edgecolor ='grey', label ="Euclidean distance") 
plt.bar(br2, norm_euclidean_accuracy, color ='g', width = barWidth, 
        edgecolor ='grey', label ="Normalized Euclidean distance") 
plt.bar(br3, cosine_accuracy, color ='b', width = barWidth, 
        edgecolor ='grey', label ="Cosine Similarity") 
   
# Adding Xticks  
plt.xlabel('Different K', fontweight ='bold') 
plt.ylabel('Accuracy in %', fontweight ='bold') 
plt.title('Accuracy plot for different K and distance matrix') 
plt.xticks([r + barWidth for r in range(len(euclidean_accuracy))], 
           K_for_NN) 
plt.legend()
print("\n|--See bar graph for results visulization")
plt.show() 
print("|-------------------------------------------------------------|")