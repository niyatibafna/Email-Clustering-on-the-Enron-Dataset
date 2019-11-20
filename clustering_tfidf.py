import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import sklearn.metrics.pairwise as sk
from sklearn.feature_extraction.text import TfidfVectorizer 
import nltk

NUMBER_OF_FILES = 517402

# NUMBER_OF_FILES_test = 1
# files_test = [""]*NUMBER_OF_FILES_test
# for i in range(NUMBER_OF_FILES_test):
# 	files_test[i] = open(str(i+1)+".", "r")
# tfidf_vectorizer_test = TfidfVectorizer(input = 'file', use_idf=True)
# tfidf_vectorizer_test_vectors=tfidf_vectorizer_test.fit_transform(files_test)
files = [""]*NUMBER_OF_FILES
content = [""]*NUMBER_OF_FILES
for i in range(NUMBER_OF_FILES):
	files[i] = open(str(i+1)+".", "r")
	text = nltk.word_tokenize(files[i].read())
	text = nltk.pos_tag(text)
	for j in range(len(text)):
		if(text[j][1].startswith("N") or text[j][1].startswith("V")):
			content[i] = content[i] + " "+ text[j][0]
	files[i].seek(0)
tfidf_vectorizer=TfidfVectorizer(input = 'content', use_idf=True)
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(content)
# print(tfidf_vectorizer_vectors.shape)
# print(type(tfidf_vectorizer_vectors))
tfidf_vectorizer_vectors.toarray()
# print(type(tfidf_vectorizer_vectors))
Kmean = KMeans(n_clusters=2)
labels = Kmean.fit_predict(tfidf_vectorizer_vectors)
centroids = Kmean.cluster_centers_
print(centroids)
print(labels)
print(Kmean.predict(tfidf_vectorizer_vectors))

features = tfidf_vectorizer.get_feature_names()

tfidf_array_format = tfidf_vectorizer_vectors.toarray()
for i in range(NUMBER_OF_FILES):
	row = tfidf_array_format[i]
	topn_ids = np.argsort(row)[::-1][:20]
	# print(topn_ids)
	# print(row[topn_ids[0]])
	top_feats = [(features[j], row[j]) for j in topn_ids]
	df = pd.DataFrame(top_feats, columns=['features', 'score'])
	print("Top 20 words and scores in document: ")
	print(df)
cosine_similarities = sk.cosine_similarity(tfidf_array_format)
print("Printing cosine cosine_similarities pairwise:")
print(cosine_similarities)
