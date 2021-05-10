'''
Name: Anand Jhunjhunwala
Roll Number: 17EC35032
Coding Assignment 2: K-Means clustering

'''
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

arglist = sys.argv

csv_file = "Mall_Customers.csv"
feature_map = {"Age":2, "Annual_Income":3, "Spending_Score":4}
colours = ['r', 'g', 'b', 'y', 'k', 'c']
def get_data(data):
	if os.path.exists(data):
		df = pd.read_csv(data)
	else:
		print("|-------------------------------------------------------------|")
		print("--The file "+data+" was not found in local folder \n--Please keep the csv file in the same folder as code")
		print("|-------------------------------------------------------------|")
	return df

def extract_features(df, feature1, feature2, feature_map):
	f1 = feature_map[feature1]
	f2 = feature_map[feature2]
	data = df.iloc[:,[f1, f2]]
	data = data.values
	return data

if(len(arglist) == 2):
	num_cluster = int(arglist[1])
	if(num_cluster >=2 and num_cluster <=6):
		print("|-------------------------------------------------------------|")
		print("|- Loading data")
		data = get_data(csv_file)
		print("|- Data loaded")
		input_combination = [["Age", "Annual_Income"], ["Annual_Income", "Spending_Score"], ["Age","Spending_Score"]]
		fig = plt.figure()
		fig.suptitle("Cluster of Clients with K=" + str(num_cluster), fontweight ="bold", fontsize=30)
		for i, c in enumerate(input_combination):
			data_f = extract_features(data, c[0], c[1], feature_map)
			kmeans = KMeans(n_clusters =num_cluster, init = 'k-means++', max_iter=400, n_init = 10, random_state=0)
			print("|- Constructing clusters with K = {0}, and features".format(num_cluster))
			print("  |-" + c[0])
			print("  |-" + c[1])
			cluster_labels = kmeans.fit_predict(data_f)
			num = (len(input_combination))*100 + 10 + (i+1)
			k = fig.add_subplot(num)
			for c_num in range(num_cluster):
				k.scatter(data_f[cluster_labels==c_num, 0], data_f[cluster_labels==c_num, 1], s =100, c = colours[c_num], label = ("Cluster "+ str(c_num)))
			k.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'm', label = 'Centroids')
			k.set_xlabel(c[0], fontweight ="bold", fontsize=15)
			k.set_ylabel(c[1], fontweight ="bold", fontsize=15)
			k.legend()
			print("|- Cluster constructed")
		print("|- See plot for the results")
		plt.show()
		print("|-------------------------------------------------------------|")
	else:
		print("|-------------------------------------------------------------|")
		print("!-Provided (K) is not of range please keep it in range [2,6]")
		print("|-------------------------------------------------------------|")

else:
	print("|-------------------------------------------------------------|")
	print("!-Please provide 1 argument i.e number of required clusters(K) in range [2,6]")
	print("|-------------------------------------------------------------|")