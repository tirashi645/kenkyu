import numpy as np
from matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

# mnistの形状28,28,1
img_col = 28
img_row = 28
channels = 1
img_shape = (img_row, img_col, channels) 