import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skfuzzy.cluster import cmeans
from sklearn.datasets import make_blobs as mb

clf = KMeans(n_clusters = 3)
N = 100

dataset = mb(centers = 3)
features = np.array(dataset[0])
#pred = clf.fit_predict(features)
cm_result = cmeans(features.T, 3, 2, 0.003, 10000)
u = cm_result[1].T

class_list = []

for i in u:
    class_list.append(np.argmax(i))

print(class_list)