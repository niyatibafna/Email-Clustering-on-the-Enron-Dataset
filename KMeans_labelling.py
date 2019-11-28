#Understanding the topics
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import sklearn.metrics.pairwise as sk
from sklearn.feature_extraction.text import TfidfVectorizer 
import nltk
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.externals import joblib
from joblib import dump
model = joblib.load("Kmeans7.model")
labels = model.labels_
NUMBER_OF_CLUSTERS = 7
categorized = [["START"] for j in range(NUMBER_OF_CLUSTERS)]
for j in range(NUMBER_OF_CLUSTERS):
	print(categorized[labels[j]])
NUMBER_OF_FILES = 51740
files = [""]*NUMBER_OF_FILES
content = [""]*NUMBER_OF_FILES
for i in range(1, NUMBER_OF_FILES):
	files[i] = open("clean/"+str(10*i)+".txt", "r")
	content[i] = files[i].read()
	print(labels[i-1])
	categorized[labels[i-1]].append(content[i])
	files[i].close()
	
for i in range(NUMBER_OF_CLUSTERS):
	new_file = open("Kmeans7_clustered.files/"+str(i)+ ".txt", "w+")
	for element in categorized[i][:50]:
		new_file.write(element)
		new_file.write("\n\n - - x - - \n\n")
	new_file.close()

content = []
for i in range(NUMBER_OF_CLUSTERS):
	text = ' '.join(categorized[i])
	content.append(text)

tfidf_vectorizer=TfidfVectorizer(input = 'content', use_idf=True)
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(content)
tfidf_vectorizer_vectors.toarray()
features = tfidf_vectorizer.get_feature_names()
# print(features[i])
tfidf_array_format = tfidf_vectorizer_vectors.toarray()

#For understanding output
for i in range(NUMBER_OF_CLUSTERS):
	new_file = open("Kmeans7_clustered.files/"+str(i)+ ".txt", "a")
	new_file.seek(0)
	row = tfidf_array_format[i]
	topn_ids = np.argsort(row)[::-1][:20]
	# print(topn_ids)
	# print(row[topn_ids[0]])
	top_feats = [(features[j], row[j]) for j in topn_ids]
	df = pd.DataFrame(top_feats, columns=['features', 'score'])
	print("Top 30 words and scores in document: ")
	print(df)
	df.to_string(new_file)
	new_file.close()
