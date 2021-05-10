'''
Name: Anand Jhunjhunwala
Roll Number: 17EC35032
Coding Assignment 1: Decision Tree

'''
import sys
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

arglist = sys.argv

# Configurable parameters
train_data = "iris_train_data.csv"  #Name of train csv file
test_data = "iris_test_data.csv"	#Name of test csv file
target_col_name = "class"			#target value column name
decision_tree_criterion = arglist[1]#default gini   
if(arglist[2]=='None'):             #default None
	max_depth = None
else:
	max_depth = int(arglist[2])

min_samples_leaf = int(arglist[3]) 	#default 1

#This function takes csv file name and return its dataframe
def get_data(data):
	if os.path.exists(data):
		df = pd.read_csv(data)
	else:
		print("|-------------------------------------------------------------|")
		print("--The file "+data+" was not found in local folder \n--Please keep the csv file in the same folder as code")
		print("|-------------------------------------------------------------|")
	return df

'''To train the decision tree target values need to be mapped to a integer
   this functions does that and return a new dataframe with a new column named
   labels which contains integer classes and also returns the list of 
   original target names and dict of this integer mapping which will be used to 
   map test data.
'''
def replace_label_train(df, target_name):
	new_df = df.copy()
	targets = new_df[target_name].unique()
	class_name_map = {}
	for i, class_name in enumerate(targets):
		class_name_map[class_name] = i
	new_df["label"] = new_df[target_name].replace(class_name_map)
	return (new_df, class_name_map, targets)

#Same function as of above but for test data
def replace_label_test(df, target_name, class_name_map):
	new_df = df.copy()
	new_df["label"] = new_df[target_name].replace(class_name_map)
	return new_df

#Function to return the accuracy of the classifier
def Find_accuracy(classifier, data, label):
	prediction = classifier.predict(data)
	correct = (prediction == label)
	total_correct = sum(correct)
	total_data = len(label)
	return (float(total_correct)/total_data)*100

#Define Classifier using parameters given in Configuration block
classifier = DecisionTreeClassifier(criterion=decision_tree_criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

#Get the training data
train_dataframe = get_data(train_data)
train_data_with_label, class_name_map, targets = replace_label_train(train_dataframe, target_col_name)
features = list(train_data_with_label.columns[:4])
labels_train = train_data_with_label["label"]
data_train = train_data_with_label[features]

#Train the classifier on training data
classifier.fit(data_train, labels_train)

#Get the test data
test_dataframe = get_data(test_data)
test_data_with_label = replace_label_test(test_dataframe, target_col_name, class_name_map)
labels_test = test_data_with_label["label"]
data_test = test_data_with_label[features]

#Find training accuracy
train_accuracy = Find_accuracy(classifier, data_train, labels_train)

#Find test accuracy
test_accuracy = Find_accuracy(classifier, data_test, labels_test)

print("|-------------------------------------------------------------|")
print("--Decision Tree trained using " + decision_tree_criterion)
print("--Parameters used:")
print("  |-max_depth = " + str(max_depth))
print("  |-min_samples_leaf = " + str(min_samples_leaf))
print("\n--Training Accuracy: " + "{:.2f}%".format(train_accuracy))
print("--Test Accuracy: " + "{:.2f}%".format(test_accuracy))
print("|-------------------------------------------------------------|")
