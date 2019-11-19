import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer 
# number_of_files_test = 1
# files_test = [""]*number_of_files_test
# for i in range(number_of_files_test):
# 	files_test[i] = open(str(i+1)+".", "r")
# tfidf_vectorizer_test = TfidfVectorizer(input = 'file', use_idf=True)
# tfidf_vectorizer_test_vectors=tfidf_vectorizer_test.fit_transform(files_test)
number_of_files = 5
files = [""]*number_of_files
for i in range(number_of_files):
	files[i] = open(str(i+1)+".", "r")
	files[i].seek(0)
# docs = [files[i].read() for i in range(number_of_files)]
tfidf_vectorizer=TfidfVectorizer(input = 'file', use_idf=True)
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(files)
print(tfidf_vectorizer_vectors.shape)
print(tfidf_vectorizer_vectors)
print(type(tfidf_vectorizer_vectors))
tfidf_vectorizer_vectors.toarray()
print(type(tfidf_vectorizer_vectors))
print(tfidf_vectorizer.get_feature_names())
# print(tfidf_vectorizer_vectors[0][7])
Kmean = KMeans(n_clusters=2)
labels = Kmean.fit_predict(tfidf_vectorizer_vectors)
centroids = Kmean.cluster_centers_
print(centroids)
print(labels)
print(Kmean.predict(tfidf_vectorizer_vectors))

for i in range(number_of_files):
	print((tfidf_vectorizer_vectors[i][2]))

