import pickle
from matplotlib import pyplot as plt

outputpath = '/media/koshiba/Data/pix2pix/output'

with open(outputpath + '/gloss.pkl', 'rb') as f:
    G_loss = pickle.load(f)
with open(outputpath + '/dloss.pkl', 'rb') as f:
    D_loss = pickle.load(f)

epochs = [i for i in range(len(D_loss) + 1)]

print(D_loss)