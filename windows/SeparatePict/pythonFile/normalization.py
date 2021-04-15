<<<<<<< HEAD
import numpy as np

def sma(x, num):
    b=np.ones(num)/num

    y2=np.convolve(x, b, mode='same')#移動平均
    
    return  y2

def sma5(x):
    b = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
    y2=np.convolve(x, b, mode='same')#移動平均

    return y2

def err(x1, x2):
    y = np.abs(x1 - x2)

    return y

def corrcoef(x1, x2):
    y = np.corrcoef(x1, x2, rowvar=True)

    return y

def min_max_normalization(x):
    x_min = min(x)
    x_max = max(x)
    x_norm = (x - x_min) / ( x_max - x_min)
=======
import numpy as np

def sma(x, num):
    b=np.ones(num)/num

    y2=np.convolve(x, b, mode='same')#移動平均
    
    return  y2

def sma5(x):
    b = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
    y2=np.convolve(x, b, mode='same')#移動平均

    return y2

def err(x1, x2):
    y = np.abs(x1 - x2)

    return y

def corrcoef(x1, x2):
    y = np.corrcoef(x1, x2, rowvar=True)

    return y

def min_max_normalization(x):
    x_min = min(x)
    x_max = max(x)
    x_norm = (x - x_min) / ( x_max - x_min)
>>>>>>> bdd2750e416964698f1ddbe1736dcfb1853f2963
    return x_norm