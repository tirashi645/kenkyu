<<<<<<< HEAD
import numpy as np

a = np.array([[[1,2,3], [2,4,6]],
     [[0,0,0], [0,0,0]],
     [[1,1,1], [9,8,7]]])

b = a.sum(axis=2)

print(b)
=======
import numpy as np

indices = (np.arange(350) // 50).astype(np.uint32).reshape(1, -1)
indices = np.broadcast_to(indices, (50, 350))
print(indices[0])
>>>>>>> bdd2750e416964698f1ddbe1736dcfb1853f2963
