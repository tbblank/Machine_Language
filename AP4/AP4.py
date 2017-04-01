from sklearn.cluster import KMeans;
import pandas as pd;
import numpy as np;
from sklearn.cross_validation import train_test_split;
from sklearn import metrics;

#import of data
data = pd.read_csv('seeds.csv');

#Separate the data from results
X = data[['area','perimeter','compactness','kernel_length','kernel_width','asymmetry','groove_length']];
Y = data[['variety']];

#x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=1);

kmeans = KMeans(n_clusters=3);
#kmeans.fit(x_train);
kmeans.fit(X);

#Show the centroid coordinates
print("Cluster centroid coordinates: ", kmeans.cluster_centers_);

#kmeans.predict(x_test);
#Score the results using several different methods
print("fowlkes_mallows_score: ", metrics.fowlkes_mallows_score(Y['variety'], kmeans.labels_));
print("adjusted_rand_score: ", metrics.adjusted_rand_score(Y['variety'], kmeans.labels_));
print("adjusted_mutual_info_score: ", metrics.adjusted_mutual_info_score(Y['variety'], kmeans.labels_));
print("homogeneity_score: ", metrics.homogeneity_score(Y['variety'], kmeans.labels_));
print("completeness_score: ", metrics.completeness_score(Y['variety'], kmeans.labels_));