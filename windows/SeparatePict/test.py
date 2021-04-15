import numpy as np

a = np.array([[[1,2,3], [2,4,6]],
     [[0,0,0], [0,0,0]],
     [[1,1,1], [9,8,7]]])

b = a.sum(axis=2)

print(b)