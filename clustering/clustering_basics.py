# Clustering Basics
import pandas as pd

data = pd.read_csv('data/Course-Hashtags.csv')
data.head()

# Separate Hashtags and title to lists
hash_list = data['HashTags'].tolist()
title_list = data['Course'].tolist()

# Do TF-IDF conversion of hashtags
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
hash_matrix = vectorizer.fit_transform(hash_list)
vectorizer.get_feature_names()[:5]

#--- Clustering TF-IDF data
# Use KMeans clustering from scikit-learn
from sklearn.cluster import KMeans

# Split data into 3 clusters
kmeans = KMeans(n_clusters=3).fit(hash_matrix)

# get Cluster labels
clusters = kmeans.labels_

for group in set(clusters):
    print('\nGroup: ', group, '\n-----------------')
    for i in data.index:
        if( clusters[i] == group):
            print(title_list[i])

#--- Finding optimal Cluster size
sosd = []

K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km=km.fit(hash_matrix)
    sosd.append(km.inertia_)

print('Sum of squared distances: ', sosd)

# Plot sosd against number of clusters
import matplotlib.pyplot as plt
plt.plot(K, sosd, 'bx-')
plt.xlabel('Cluster count')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method for Optimal Cluster Size')
plt.show()
