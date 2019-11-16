import pandas as pd
import numpy as np
import copy
n=5
files = [""]*n
for i in range(n):
	files[i] = open(str(i+1)+".", "r")
dictionary = {}
index = 0
for i in range(n):
	email_text = files[i].read().split()
	files[i].seek(0)
	print(email_text)
document_term_matrix = []
for i in range(n):
	email_text = files[i].read().split()
	files[i].seek(0)
	print(email_text)
	for word in email_text:
		if(word not in dictionary):
			dictionary[word] = index
			index += 1
print(dictionary)
vocabulary = len(dictionary)
document_term_matrix = np.zeros((n, vocabulary))
for i in range(n):
	email_text = files[i].read().split()
	print(email_text)
	# document_row = np.zeros((vocabulary))
	for word in email_text:
		print(dictionary[word])
		document_term_matrix[i][dictionary[word]] += 1
		# document_row[dictionary[word]] += 1
	# np.append(document_term_matrix, document_row)
	# document_term_matrix.append(document_row)


print((document_term_matrix[1]))
from sklearn.cluster import KMeans
Kmean = KMeans(n_clusters=2)
Kmean.fit(document_term_matrix)
centroids = Kmean.cluster_centers_
print(centroids)
