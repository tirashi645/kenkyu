import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skfuzzy.cluster import cmeans
from sklearn.datasets import make_blobs as mb

li = np.array([[[1,2,3],[10,20,30]],[[12,22,32],[102,202,302]]])

li = li * 10
li = (li - 5) / 5 * 255
print(li)