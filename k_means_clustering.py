# import the necessary packages
from __future__ import print_function
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from scipy.stats import mode
from sklearn.metrics import accuracy_score
import numpy as np
import sklearn
 
# handle older versions of sklearn
if int((sklearn.__version__).split(".")[1]) < 18:
	from sklearn.cross_validation import train_test_split
 
# otherwise we're using at lease version 0.18
else:
	from sklearn.model_selection import train_test_split
 
# load data

data = load_digits()
data.data.shape

# Clustering 10 digits (0-9) based on digits image data
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(data.data)
kmeans.cluster_centers_.shape

# Show the 10 clusters
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    
# Marching predicted clusters with correct digits (most common value for a cluster)
# saved as labels
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(data.target[mask])[0]

# Accuracy of Clustering by comparing with the right targets
accuracy_score(data.target, labels)
# 0.7935447968836951