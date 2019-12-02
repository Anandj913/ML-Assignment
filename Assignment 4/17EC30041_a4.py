#---------------------------------------------------------------------------------------------------------------------
'''
Roll_number = 17EC30041
Name = Anand Jhunjhunwala
Assignment Number = 4 (K-Means Clusters)
Specific compilation instruction : Put data4_19.csv file in the same folder in which this code is placed
								   run python 17EC30041_a4.py
'''
# ---------------------------------------------------------------------------------------------------------------------

import csv
import math 
import os
from random import sample


K = 3 #K to be used for k means
Iterations = 10
f_to_use = 4 #Feature number upto which distance will be calculated starting from 1 

features = ['sepal_length', 'sepal_width' , 'petal_length' , 'petal_width', 'clusters']
clusters_name = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


#Read data from csv
def convert_csv(filename, features):  

	filepath = os.path.join(os.getcwd(), filename)
	with open(filepath, 'rt') as file:
		f = csv.reader(file)
		total_data = []
		for d in f:
			total_data.append(d)

		total_d = len(total_data)
		feature = features
		feature_map = {}
		idx_map = {}
		for i in range(0, len(feature)):
			feature_map[feature[i]] = i
			idx_map[i] = feature[i]
		data = {
		'feature': feature,
		'data': total_data[0:total_d],
		'feature_map': feature_map,
		'idx_map': idx_map
		}
		return data;

#To calculate euclidean distance between two points
def distance(data1, data2):
	dist = 0.0
	for i in range(0,f_to_use):
		dist += (float(data1[i]) - float(data2[i]))**2
	return math.sqrt(dist)

#To initalize the first clusters center
def initalize(data , k):
	r_idx = sample(range(len(data)), k)
	clusters = []
	for i in r_idx:
		clusters.append(data[i][:])
	return clusters

#To seprate points to their specific clusters 	
def Clusteridentification(data , k, clusters):
	clustersidx = []
	for i in range(len(data)):
		min_dis = 1000000
		for n in range(len(clusters)):
			d = distance(data[i], clusters[n])
			if d < min_dis:
				min_dis = d
				temp = n
		clustersidx.append(temp)

	return clustersidx

#Determine new centre of clusters
def new_clusters(data, c_idx, k):
	new_cluster = list()
	for i in range(k):
		data_sep = list()
		for idx , val in enumerate(c_idx):
			if(val == i):
				data_sep.append(data[idx])
		cluster = list()
		for j in range(f_to_use):
			s = 0.0
			for d in data_sep:
				s = s + float(d[j])
			cluster.append(str(s/len(data_sep)))
		new_cluster.append(cluster)
	return new_cluster

#Algo 
def Kmean(data, k):
	cluster_center = initalize(data, k)

	for i in range(Iterations):
		cluster_labels = Clusteridentification(data , k , cluster_center)
		cluster_center = list()
		cluster_center = new_clusters(data, cluster_labels, k)

	cluster_labels = Clusteridentification(data, k, cluster_center)

	return cluster_labels , cluster_center

#Seprate data into clusters as identified by algo
def dataidx_sepration(labels, k):
	cluster_idx = list()
	for i in range(k):
		temp = list()
		for idx, val in enumerate(labels):
			if(val == i):
				temp.append(idx)
		cluster_idx.append(temp)
	return cluster_idx

#Original data sepration according to cluster label
def data_sepration(data, cluster_name, k):
	data_idx = list()
	for i in range(len(cluster_name)):
		temp = list()
		for idx, val in enumerate(data):
			if(val[-1] == cluster_name[i]):
				temp.append(idx)
		data_idx.append(temp)
	return data_idx

#To find Union of list
def Union(list1 , list2):
	final_list = list(set(list1) | set(list2)) 
	return len(final_list)

#To find Intersection of list
def intersection(list1, list2): 
    final = list(set(list1) & set(list2)) 
    return len(final)

#Determine Jacquard distance of each cluster w.r.t each real cluster
def Jacquard_distance(data_cluster, data_real, k):
	Jac = list()
	for i in range(k):
		temp = list()
		for j in range(len(data_real)):
			temp2 = 1- round(float(intersection(data_cluster[i],data_real[j]))/Union(data_cluster[i],data_real[j]),3)
			temp.append(temp2)
		Jac.append(temp)
	return Jac

#To print the final structure 
def Final_print(center, Jac, k , data_idx):
	print('|----------------------|K-Means Clustering Algorithm|----------------------|')
	print('| Value of K used: %d')%(K)
	print('| Number of Iteration: %d')%(Iterations)
	print('|---------------------------|Printing Clusters|----------------------------|')
	Total = 0.0
	for i in range(k):
		print('| Cluster %d')%(i)
		print('| Mean:'), 
		print(center[i])
		print('| Total element in a cluster: %d')%(len(data_idx[i]))
		print('| Jacquard Distance w.r.t each cluster:'),
		print(Jac[i])
		print('| Label: ' + clusters_name[Jac[i].index(min(Jac[i]))]) #label with minimum jac_distance
		Total = Total + min(Jac[i])
		print('|--------------------------------------------------------------------------|')
	print('| Total Jacquard distance = %.3f')%(Total)

#Main loop to call every function
def main():
	data = convert_csv('data4_19.csv', features)
	labels, center = Kmean(data['data'], K)
	dataidx_cluster = dataidx_sepration(labels, K)
	data_sep = data_sepration(data['data'], clusters_name, K)
	J_distance = Jacquard_distance(dataidx_cluster, data_sep, K)
	Final_print(center, J_distance, K, dataidx_cluster)
	
if __name__ == "__main__": main()