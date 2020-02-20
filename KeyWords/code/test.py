import csv
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
from gensim.test.utils import common_texts
import gensim.downloader as api
import nltk
import os.path
import string
import numpy as np
from sklearn import cluster
from sklearn import metrics

from itertools import groupby

def readfile(fileIN):
	nameList = []
	with open(fileIN, encoding='utf-8-sig') as csv_file:  # reading csv file for name and making a list of skill
	    csv_reader = csv.reader(csv_file)
	    next(csv_reader)
	    for row in csv_reader:
	        nameList.append(row)
	return nameList

def readfile_arg(fileIN, start, end):
	nameList = []
	with open(fileIN, encoding='utf-8-sig') as csv_file:  # reading csv file for name and making a list of skill
	    csv_reader = csv.reader(csv_file)
	    next(csv_reader)
	    for row in csv_reader:
	        nameList.append(row[start:end])
	return nameList
# function to return key for any value 
def get_key(val, my_dict): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key 
  
    return "key doesn't exist"

keyword = [] 
busi_keyword = []
cbow = []
keyword = readfile("Busi_Keyword.csv")
reviews = readfile_arg("buss1_copy.csv",0,2)

d = {}
dict_kW = dict()
with open("Busi_Keyword.csv") as f:
    for line in f:
    	(key, val) = line.split(",")
    	val =val.strip("\n")
    	if key not in d.keys():
    		d.setdefault(key,[])
    		d[key].append(val)
    	else:
    		d.setdefault(key,[])
    		d[key].append(val)
print(d)

no_punt = [] 
for review in reviews:
	review = review[1].lower()
	review = review.split()
	table = str.maketrans('', '', string.punctuation)
	stripped = [w.translate(table) for w in review]
	no_punt.append(stripped)

word_vec = api.load("glove-wiki-gigaword-100")

for review in no_punt:
	n = 0
	loop_break = False

	for value in d.values():
		if loop_break: break
		for word in review:
			if loop_break: break
			for v in value:
				try:
					sim = word_vec.similarity(word,v)
					#print(word, v, sim)
					if sim > 0.90:
					    n += 1
					    if n > 10:
					        print(review, get_key(value, d))
					        print()
					        loop_break = True
					        break



				except KeyError:
				    pass

# for review in no_punt:
# 	for value in d.values():
# 		for word in review:
# 			for v in value:
# 				try:
# 					# print(word, value)
# 					sim=float(model.wv.similarity(v,word))
# 					print(word,v,sim)
# 				except KeyError:
# 				    pass
				

# print(len(no_punt[0]))
# value = [i for i in d.values()]
# print(len(value[0]))
# r_size = len(no_punt[0])
# v_size = len(value[0])

# arr = np.zeros((v_size,r_size))
# print(arr)
# for value in d.values():
# 	print(value)
# for values_list in d.values():
	# print(len(values_list))
	# for value in values_list:
		# print(len(value))
	

# 	for word in review:
# sum_temp = 0
# sum_final = 0
# lst_sum = []
# lst_CLASS = []
# model = Word2Vec(no_punt, min_count=1)
# for review in no_punt:
# 	review_size = len(review)
# 	print("NEw Review")
# 	for value_list in d.values():
# 		print("New Class!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# 		val_size = len(value_list)
# 		arr = np.zeros((val_size,review_size))
# 		for value in value_list:
# 			row = 0
# 			for word in review:
# 				col = 0
# 				try:
# 					sim=float(model.wv.similarity(word,value))
# 					print(word,value,sim)
# 				except KeyError:
# 				    pass
# 				else:
# 					if sim > 0.70:
# 						arr[row][col] = 1
# 				col = col+1
# 			row = row+1
# 		print(arr)




	# 		for x in value_list:
	# 			row = 0
	# 			for word in review:
	# 				col = 0
					
	# 				col = col +1 
	# 			row = row+1
	# 		counter = 0
	# 		lst = []
	# 		sum_temp = 0
	# 		# n = 0
	# 		for a in range(0,arr.shape[0]):
	# 			for b in range(0,arr.shape[1]):
	# 				if arr[a][b] == 1:
	# 					lst.append(1)
	# 			# n = n+1
	# 		sum_temp = sum(lst)
	# 		print(sum_temp)
	# 		final_kW = value_list
			
	# 		if sum_temp > sum_final:
	# 			for cLass, value_kW_list in d.items():
	# 				if value_kW_list == value_list:
	# 					final_class = cLass  
	# 			sum_final = sum_temp
	# 		if sum_temp == 0:
	# 			sum_final = 0
	# 			final_class = 0
	# 	# if final_class != 0:
	# 	lst_sum.append(sum_final)
	# lst_CLASS.append(final_class)
	# print(review, final_class)
	
# print(lst_sum)
# for review in no_punt:
# 	for value in d.values():
# 		r_size = len(review)
# 		for i in value:
# 			v_size = int(i)
# 			arr = np.zeros((v_size, r_size))
# 			for word in review:
# 				for i in range(0, v_size):


# for x in range(0, arr.shape[0]):
# 	for y in range(0, arr.shape[1]):
# 		print(a[x,y])
# for review in no_punt:
# 	for value in d.values():
# 		arr = np.zeros(len(review),len(value))
# 		print(len(review), len(value))
# 		for word in review:
# 			try:
# 				sim=float(model.wv.similarity(kW,word))
# 			except KeyError:
# 				pass
# 			else:
# 				if sim > 0.80:
# 					kW_arr[row][col] = 1







		# for row in range(arr.shape[0]):
		# 	for col in range(arr,shape[1]):
		# 		for word in review:
					

		# for row in range(arr.shape[0]):
		# 	for col in range()

# 	for word in review:
# 		for classs in d.keys():
# 			for
# 	review_size = len(review)

# 	for word in review:
# 		for value in d.values():
# 			value_size = len(value)
			

# for value in d.values():
# 	print(len(value))

# model = Word2Vec(no_punt, min_count=1)

# for value in d.values():
# 	for review in no_punt:
# 		value_size = len(value)
# 		review_size = len(review)
# 		kW_arr = np.zeros((value_size,review_size))
# 		for word in review:
# 			try:
# 			    sim=float(model.wv.similarity(kW,word))
# 			except KeyError:
# 			    pass
# 			else:
# 				if sim > 0.80:
# 					kW_arr[value][word] = 1





# 			try:
# 			    sim=float(model.wv.similarity(kW,word))
# 			except KeyError:
# 			    pass
# 			else:
			    
# 			    	kW_arr[]
# 			        print(kW, word, model.wv.similarity(kW,word))

# print(dict_kW)
# cbow = [i[0:2] for i in keyword]
# # dict_kW = {i[0]: i[1] for i in cbow}
# print(cbow,"\n")
# # print(cbow[1][1])
# # print(dict_kW)
# threshold = 0.80
# no_punt = [] 

# for review in reviews:
# 	review = review[0].split()
# 	table = str.maketrans('', '', string.punctuation)
# 	stripped = [w.translate(table) for w in review]
# 	no_punt.append(stripped)


# cbow = cbow.strip()
# print(cbow)


# for line in cbow:
# 	if line[0] in dict_kW:
# 		dict_kW[line[0].append(line[1])]
# 	else:
# 		dict_kW[line[0]] = [line[1]]
# print(dict_kW)

# for i in keyword:
# 	element = i
# 	count = i.count(i[0])
# 	count_lst.append(count)
	# if i[0] not in lst:
	# 	count = 0
	# 	count = count+1
	# 	lst.append(i[0])
	# 	count_lst.append(count)
	# elif i[0] in lst:
	# 	count = count+1
		# count_lst.append(count)
# print(lst)
# print(count_lst)
# size = int(len(no_punt))


# print(no_punt[0])
# print(len(no_punt[0]))
# # print(cbow[0])
# key = []
# value = []
# lst = []
# for i in cbow:
# 	print("iTh value: ", i)
# 	if len(lst) == 0:
# 		lst.append(i[0])
# 		value.append(i[1])
# 	else:
# 		if i[0] not in lst:
# 			lst.append(i[0])
# 		else:
# 			value.append(i[1])

# 	print("list", lst)
# 	print("value", value)
# 	if i in lst[0]
# 	key.append(i[0][1])
# 	value.append(i[0][1])
# print(key)
# print(value)

# model = Word2Vec(no_punt, min_count=1)
# for review in no_punt:
# 	review_size = len(review)
# 	for word in review:
# 		for kW in cbow[:1]:
# 			cbow
# 			try:
# 			    sim=float(model.wv.similarity(kW,word))
# 			except KeyError:
# 			    pass
# 			else:
# 			    if sim > 0.80:
# 			        print(kW, word, model.wv.similarity(kW,word))
			    
			# try:
			# 	if model.wv.similarity(kW,word) > 0.80:
			# 		print(kW, word, model.wv.similarity(kW,word))
			# except KeyError:
			# 	print("not in vocabulary")
			# if model.wv.similarity(kW,word) > 0.80:
			# 	print(kW, word, model.wv.similarity(kW,word))
			# else:
			# 	print("not in vocabulary")
			# if kW in no_punt:
			


# for cbow_i in cbow:
# 	for sentence in no_punt:
# 		words = sentence[0].split()
# 		# print("Words:",words)
# 		for word in words:
# 			# if model.wv.similarity(cbow_i,word) > 0.8:
# 			# print("word:", word)
# 			print(model.wv.similarity(cbow_i,word))
# 		# print(similarity)





# print(reviews[1])




# sentences=[]
# with open("model.csv", encoding='utf-8-sig') as csv_file:
#  	csv_reader = csv.reader(csv_file)
#  	for row in csv_reader:

#  		sentences.append(row)

# print(sentences)
 
 
# # training model
# model = Word2Vec(sentences, min_count=1)
 
# # get vector data
# X = model.wv[model.wv.vocab]
# print (X)
 
# print (model.wv.similarity('this', 'is'))
 
# print (model.wv.similarity('post', 'book'))
 
# print (model.wv.most_similar(positive=['machine'], negative=[], topn=2))
 
# print (model.wv['the'])
 
# print (list(model.wv.vocab))
 
# print (len(list(model.wv.vocab)))
 
 
 
 
# NUM_CLUSTERS=3
# kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
# assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
# print (assigned_clusters)
 
# words = list(model.wv.vocab)
# for i, word in enumerate(words):  
#     print (word + ":" + str(assigned_clusters[i]))
 
 
 
# kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
# kmeans.fit(X)
 
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
 
# print ("Cluster id labels for inputted data")
# print (labels)
# print ("Centroids data")
# print (centroids)
 
# print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
# print (kmeans.score(X))
 
# silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
 
# print ("Silhouette_score: ")
# print (silhouette_score)