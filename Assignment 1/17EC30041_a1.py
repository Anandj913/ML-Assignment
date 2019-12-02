#---------------------------------------------------------------------------------------------------------------------
# Roll_number = 17EC30041
# Name = Anand Jhunjhunwala
# Assignment Number = 1 (Decision Trees)
# Specific compilation instruction : Put data1_19.csv file in the same folder in which this code is placed and run python 17EC30041_a1.py
# ---------------------------------------------------------------------------------------------------------------------

import csv
import math
import os

test_set_proportion = 0.0
output_feature = 'survived'
def convert_csv(filename, test_proportion, status): # it read csv file and store the training and test set depending on status

	filepath = os.path.join(os.getcwd(), filename)
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
		total_train = int((total_d-1)*(1-test_proportion))
		if(status == 1):
			data = {
			'feature': feature,
			'data': total_data[1:total_train],
			'feature_map': feature_map,
			'idx_map': idx_map
			}
		else:
			data = {
			'feature': feature,
			'data': total_data[(total_train+1):],
			'feature_map': feature_map,
			'idx_map': idx_map
			}
		return data;
def find_values_of_feature(data): # it return the features with its all unique values
	val_col = {}
	for feature in data['feature']:
		val_col[feature] = set()
		for i in range(0, len(data['data'])):
			value = data['data'][i][data['feature_map'][feature]] 
			if value not in val_col.keys():
				val_col[feature].add(value)
			
	return val_col

def get_feature_val_count(data,feature): # it count the total number of all diffrent values of features.
	index = data['feature_map'][feature]
	val_feature = {}
	for r in data['data']:
		value = r[index]
		if value in val_feature:
			val_feature[value] = val_feature[value] + 1
		else:
			val_feature[value] = 1
	return val_feature

def entropy(num, feature_val_count): #calculate the entropy of the feature value map provided as input.
	ent = 0 
	for val in feature_val_count.keys():
		p = feature_val_count[val]/float(num)
		ent += -p*math.log(p,2) 
	return ent

def partition_on_feture_val(data , feature): #split the data given into diffrent feature value of feature provided in input.
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

def avg_entropy(data, split_feature, output_feature): #calculate the avg entropy of feature provided as input
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

def feature_maxed_value(value_count_of_feature): #return the feature with max value  
	max_value = 0
	max_feature = None
	for feature_value in value_count_of_feature.keys():
		if(value_count_of_feature[feature_value] > max_value):
			max_value = value_count_of_feature[feature_value]
			max_feature = feature_value
	return max_feature

def Decision_tree(data, feature_val_map, input_features, output_feature,store): #recursive function to build the tree 

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
def tree_print(root,store): # it print the function in required format
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
	count = 0
	total = float(len(data))
	for i , L in enumerate(data):
		if(L[-1] == predict[i]):
			count = count + 1
	accuracy = (count/total)*100
	print(' Accuracy on traning set = %.2f')%(accuracy)
	print('|--------------------------------------------------------------------|')

def predict(data, main_node):
	predicted = []
	for D in data['data']:
		tree_node = main_node
		if 'label' in tree_node:
			predicted.append(tree_node['label'])
			continue
		else:
			feature = tree_node['feature']
			tree_node = tree_node['nodes']
			tree_node = tree_node[D[data['feature_map'][feature]]]
		if 'label' in tree_node:
			predicted.append(tree_node['label'])
			continue
		else:
			feature = tree_node['feature']
			tree_node = tree_node['nodes']
			tree_node = tree_node[D[data['feature_map'][feature]]]
		if 'label' in tree_node:
			predicted.append(tree_node['label'])
			continue
		else:
			feature = tree_node['feature']
			tree_node = tree_node['nodes']
			tree_node = tree_node[D[data['feature_map'][feature]]]
		if 'label' in tree_node:
			predicted.append(tree_node['label'])
			continue
		else:
			feature = tree_node['feature']
			tree_node = tree_node['nodes']
			tree_node = tree_node[D[data['feature_map'][feature]]]
	return predicted
def main():
	store = list()
	data_train = convert_csv('data3_19.csv', test_set_proportion, 1) # last argument will be 1 for training set and 0 for test set.
	data_test = convert_csv('test3_19.csv', test_set_proportion, 0)
	input_features= set(data_train['feature']) # set of all feature value here(pclass, gender, age, survived)
	input_features.remove(output_feature) # trainable feature value (pclass, gender, age)
	feature_val_map = find_values_of_feature(data_train) # return value of diffrent features. 
	top_node = Decision_tree(data_train, feature_val_map, input_features, output_feature, store) # buid the tree
	P = predict(data_train, top_node)
	tree_print(top_node,store) # print the tree
	accuracy(P, data_train['data']) #print accuracy on traning data 

if __name__ == "__main__": main()
