

## Also, chooses #of clusters automatically.
####

'''
Unlike the mean shift algorithm, the DBSCAN algorithm is both 
highly scalable and makes no assumptions about the underlying 
shape of clusters in the dataset.
'''

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=1.2, min_samples=30)
# predefined data
dbscan.fit(data)

# cluster assignments
print('{}\n'.format(repr(dbscan.labels_)))

# core samples
print('{}\n'.format(repr(dbscan.core_sample_indices_)))
num_core_samples = len(dbscan.core_sample_indices_)
print('Num core samples: {}\n'.format(num_core_samples))

