# -*- coding: utf-8 -*-
 
 
import csv
from gensim.models import Word2Vec
 
from nltk.cluster import KMeansClusterer
import nltk
 
 
from sklearn import cluster
from sklearn import metrics
 
# training data
print("start:....")
sentences=[]
with open("model.csv", encoding='utf-8-sig') as csv_file:
 	csv_reader = csv.reader(csv_file)
 	for row in csv_reader:

 		sentences.append(row)

print(sentences)
 
 
# training model
model = Word2Vec(sentences, min_count=1)
 
# get vector data
X = model.wv[model.wv.vocab]
print (X)
 
print (model.wv.similarity('this', 'is'))
 
print (model.wv.similarity('post', 'book'))
 
print (model.wv.most_similar(positive=['machine'], negative=[], topn=2))
 
print (model.wv['the'])
 
print (list(model.wv.vocab))
 
print (len(list(model.wv.vocab)))
 
 
 
 
NUM_CLUSTERS=3
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print (assigned_clusters)
 
words = list(model.wv.vocab)
for i, word in enumerate(words):  
    print (word + ":" + str(assigned_clusters[i]))
 
 
 
kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)
 
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
 
print ("Cluster id labels for inputted data")
print (labels)
print ("Centroids data")
print (centroids)
 
print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print (kmeans.score(X))
 
silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
 
print ("Silhouette_score: ")
print (silhouette_score)