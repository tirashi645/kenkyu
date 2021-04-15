import numpy as np

indices = (np.arange(350) // 50).astype(np.uint32).reshape(1, -1)
indices = np.broadcast_to(indices, (50, 350))
print(indices[0])