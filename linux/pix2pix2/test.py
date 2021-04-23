import numpy as np

li = np.array([[[[0, 0, 0] for i in range(720)] for j in range(1280)]])
print(li.shape)
li2 = li.reshape([-1, 256, 256, 3])
print(li2.shape)