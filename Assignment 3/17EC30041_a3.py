'''
Name: Anand Jhunjhunwala
Roll Number: 17EC30041
Assignment 3: Adaboost on decision tree
Instruction: Keep train3_19.csv and test3_19.csv in the same folder as of this code and run python 17EC300041_a3.py
'''
import csv
import math
import os
import numpy as np 
from numpy.random import rand
from numpy import cumsum, sort, sum, searchsorted
rounds = 3#number of rounds to run adaboost
break_factor = 1 #fraction of dataset to take for training 
output_feature = 'survived' #Feature to predict

''' Decision Tree functions from previous assignment'''
def convert_csv(filename_train, filename_test): 
# it read csv file and store the training and test set structure

	filepath = os.path.join(os.getcwd(), filename_train)
	filepath_test = os.path.join(os.getcwd(), filename_test)
	#Read train file
	with open(filepath, 'rt') as file:
		f = csv.reader(file)
		total_data = []
		for d in f:
			total_data.append(d)

		total_d = len(total_data)
		feature = total_data[0]
		feature_map = {}
		idx_map = {}
		for i in range(0, len(feature)):
			feature_map[feature[i]] = i
			idx_map[i] = feature[i]

		data = {
		'feature': feature,
		'data': total_data[1:total_d],
		'feature_map': feature_map,
		'idx_map': idx_map
		}
	#Read test file
	with open(filepath_test, 'rt') as file_test:
		f_test = csv.reader(file_test)
		total_test = []
		for d in f_test:
			total_test.append(d)
		total_test_d = len(total_test)
		
		data_test = {
		'feature': feature,
		'data': total_test[0:total_test_d],
		'feature_map': feature_map,
		'idx_map': idx_map
		}
	return data, data_test
def find_values_of_feature(data): 
# it return the features with its all unique values
	val_col = {}
	for feature in data['feature']:
		val_col[feature] = set()
		for i in range(0, len(data['data'])):
			value = data['data'][i][data['feature_map'][feature]] 
			if value not in val_col.keys():
				val_col[feature].add(value)
			
	return val_col

def get_feature_val_count(data,feature): 
# it count the total number of all diffrent values of features.
	index = data['feature_map'][feature]
	val_feature = {}
	for r in data['data']:
		value = r[index]
		if value in val_feature:
			val_feature[value] = val_feature[value] + 1
		else:
			val_feature[value] = 1
	return val_feature

def entropy(num, feature_val_count): 
#calculate the entropy of the feature value map provided as input.
	ent = 0 
	for val in feature_val_count.keys():
		p = feature_val_count[val]/float(num)
		ent += -p*math.log(p,2) 
	return ent

def partition_on_feture_val(data , feature): 
#split the data given into diffrent feature value of feature provided in input.
	partition = {}
	partition_feature_idx = data['feature_map'][feature]
	for r in data['data']:
		feature_val = r[partition_feature_idx]
		if feature_val not in partition.keys():
			partition[feature_val] = {
			'data' : list(),
			'feature_map': data['feature_map'],
			'idx_map': data['idx_map']
			}
		partition[feature_val]['data'].append(r)
	return partition

def avg_entropy(data, split_feature, output_feature): 
#calculate the avg entropy of feature provided as input
	n = float(len(data['data']))
	partitions = partition_on_feture_val(data, split_feature)

	avg_ent = 0

	for partition_values in partitions.keys():
		partitioned_data = partitions[partition_values]
		partition_n = len(partitioned_data['data'])
		partition_labels = get_feature_val_count(partitioned_data, output_feature)
		partition_entropy = entropy(partition_n, partition_labels)
		avg_ent += (partition_n / n) * partition_entropy

	return avg_ent, partitions

def feature_maxed_value(value_count_of_feature): 
#return the feature with max value  
	max_value = 0
	max_feature = None
	for feature_value in value_count_of_feature.keys():
		if(value_count_of_feature[feature_value] > max_value):
			max_value = value_count_of_feature[feature_value]
			max_feature = feature_value
	return max_feature

def Decision_tree(data, feature_val_map, input_features, output_feature,store): 
#recursive function to build the tree 

	value_count_of_OF = get_feature_val_count(data, output_feature)
	node = {}

	if len(value_count_of_OF.keys()) == 1: #i.e leaf node
		node['label'] = feature_maxed_value(value_count_of_OF)
		return node

	if len(input_features) == 0: # i.e no more feature to check 
		node['label'] = feature_maxed_value(value_count_of_OF)
		return node

	n = len(data['data'])
	ent = entropy(n, value_count_of_OF)

	max_gain = None
	max_gain_feature = None
	max_gain_partition = None 

	for input_feature in input_features:
		avg_ent , partition = avg_entropy(data, input_feature, output_feature)
		info_gain = ent - avg_ent
		if max_gain is None or info_gain > max_gain:
			max_gain = info_gain
			max_gain_feature = input_feature
			max_gain_partition = partition

	if max_gain is None:
		node['label'] = feature_maxed_value(value_count_of_OF)
		return node
	if max_gain_feature not in store:
		store.append(max_gain_feature)
	
	node['feature'] = max_gain_feature
	node['nodes'] = {}

	remaining_input_feature = set(input_features)
	remaining_input_feature.discard(max_gain_feature)

	max_feature_values = feature_val_map[max_gain_feature]

	for feature_val in max_feature_values:
		if feature_val not in max_gain_partition.keys():
			continue
		partitions = max_gain_partition[feature_val]
		node['nodes'][feature_val] = Decision_tree(partitions, feature_val_map, remaining_input_feature, output_feature, store)
	return node 
def tree_print(root,store): 
# it print the decision tree in required format
	string = list()
	def print_format(node, string,store):
		if 'label' in node:
			string.append(':')
			string.append(node['label'])
		elif 'feature' in node:
			i = store.index(node['feature'])
			for val in node['nodes'].keys():
				string.append('\n')
				for j in range(0,i):
					string.append('\t')
				string.append('|')
				string.append(node['feature'])
				string.append('=')  
				string.append(val)
				print_format(node['nodes'][val], string,store)

	print_format(root, string, store)
	print('|---------------|' + 'Decision Tree using Information Gain' + '|---------------|')
	print(''.join(string))
	print('|--------------------------------------------------------------------|')

def accuracy(predict, data):
#Print accuracy of data provided 
	count = 0
	total = float(len(data))
	for i , L in enumerate(data):
		if(L[-1] == predict[i]):
			count = count + 1
	accuracy = (count/total)*100
	print('|--------------------------------------------------------------------|')
	print(' Accuracy on test set = %.2f')%(accuracy)
	print('|--------------------------------------------------------------------|')

def predict(data, main_node):
#Return predicted lable of the data from decision tree  
	predicted = []
	for D in data['data']:
		tree_node = main_node
		if 'label' in tree_node:
			predicted.append(tree_node['label'])
			continue
		else:
			feature = tree_node['feature']
			tree_node = tree_node['nodes']
			next_feature = D[data['feature_map'][feature]]
			if next_feature in tree_node:
				tree_node = tree_node[next_feature]
			else:
				predicted.append('yes')
				continue
			
		if 'label' in tree_node:
			predicted.append(tree_node['label'])
			continue
		else:
			feature = tree_node['feature']
			tree_node = tree_node['nodes']
			next_feature = D[data['feature_map'][feature]]
			if next_feature in tree_node:
				tree_node = tree_node[next_feature]
			else:
				predicted.append('yes')
				continue
			
		if 'label' in tree_node:
			predicted.append(tree_node['label'])
			continue
		else:
			feature = tree_node['feature']
			tree_node = tree_node['nodes']
			next_feature = D[data['feature_map'][feature]]
			if next_feature in tree_node:
				tree_node = tree_node[next_feature]
			else:
				predicted.append('yes')
				continue
			
		if 'label' in tree_node:
			predicted.append(tree_node['label'])
			continue
		else:
			feature = tree_node['feature']
			tree_node = tree_node['nodes']
			next_feature = D[data['feature_map'][feature]]
			if next_feature in tree_node:
				tree_node = tree_node[next_feature]
			else:
				predicted.append('yes')
				continue
			
	return predicted

''' -------------------------------------------------------------------------------------'''

''' ---------------------------Adaboost_code---------------------------------------------'''
def adaboost(train_data, rounds, feature_val_map, input_features):

	data = train_data['data']
	total = len(train_data['data'])
	classifiers = []
	stages = [] #store stage values of classifiers

	weights = np.ones(total)*1.0/total #initial weights of data 
	for r in range(rounds):
		random_pickup = list(range(total))
		random_pickup = np.asarray(random_pickup)
		random_pickup = np.random.choice(random_pickup, int(break_factor*total), p= weights) #pick data according to weights 
		random_pickup = random_pickup.tolist()
		resampled_data = {}
		resampled_set = []
		for i in range(int(break_factor*total)):
			resampled_set.append(data[random_pickup[i]])
		resampled_data = {
			'feature': train_data['feature'],
			'data': resampled_set,
			'feature_map': train_data['feature_map'],
			'idx_map': train_data['idx_map']
			}
		store = []
		root = Decision_tree(resampled_data, feature_val_map, input_features, output_feature, store) #build decision tree on sampled data
		# tree_print(root, store)
		prediction = predict(train_data, root) #prediction from decision tree learned 

		error = 0.0
		for i in range(len(prediction)):
			error += (prediction[i] != data[i][-1])*weights[i]
		stage = np.log((1-error)/error) #calculation of stage value

		stages.append(stage) #store stage value
		classifiers.append(root) #store classifier

		for i in range(total): #updates weights
			y = data[i][-1]
			h = prediction[i]
			k = (1 if h ==y else -1)
			weights[i] = weights[i] * np.exp(stage*k)

		sum_weights = sum(weights)
		normalized_weights = [float(w) / sum_weights for w in weights]
		weights = normalized_weights

	return classifiers, stages 

def classify_by_adaboost(root_set , weights_set , example):
#Return prediction using adaboost learned classifiers 
	classification = np.zeros(len(example['data']))
	remap = []
	for i , R in enumerate(root_set):
		prediction_by_weekclassifier = predict(example, R)
		# print(prediction_by_weekclassifier[:60])
		mapping = []
		for k in prediction_by_weekclassifier: #Mapping yes to 1 and no to -1
			if(k == 'yes'):
				mapping.append(1)
			else:
				mapping.append(-1)
		mapping = np.asarray(mapping)
		mapping = weights_set[i]*mapping
		classification += mapping
	classification = classification.tolist()
	for k in classification:
		if(k >= 0.0):
			remap.append('yes')
		else:
			remap.append('no')

	return remap

def main():
	data_train, data_test = convert_csv('data3_19.csv', 'test3_19.csv') #prepare test and train data 
	input_features= set(data_train['feature']) # set of all feature value here(pclass, gender, age, survived)
	input_features.remove(output_feature) # trainable feature value (pclass, gender, age)
	feature_val_map = find_values_of_feature(data_train) # return value of diffrent features. 
	C, W = adaboost(data_train, rounds, feature_val_map, input_features) #Run adaboost
	P = classify_by_adaboost(C, W, data_test) # Classify using adaboost
	accuracy(P, data_test['data']) # print accuracy 
if __name__ == "__main__": main()