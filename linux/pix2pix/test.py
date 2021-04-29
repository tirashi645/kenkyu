import numpy as np

X_raw = np.array([[[1,2,3], [2,3,4],[3,4,5]], 
                [[9,8,7],[8,7,6],[7,6,5]],
                [[1,2,3], [2,3,4],[3,4,5]], 
                [[9,8,7],[8,7,6],[7,6,5]]])
print(X_raw,'\n---------------------')
for i in range(len(X_raw)):
    X_raw[i, :, 0], X_raw[i, :, 2] = X_raw[i, :, 2], X_raw[i, :, 0].copy()
print(X_raw)