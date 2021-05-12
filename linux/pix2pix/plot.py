import pickle
import matplotlib.pyplot as plt

outputpath = '/media/koshiba/Data/pix2pix/output'

with open(outputpath + '/gloss.pkl', 'rb') as f:
    G_loss = pickle.load(f)
with open(outputpath + '/dloss.pkl', 'rb') as f:
    D_loss = pickle.load(f)

epochs = [i+1 for i in range(len(D_loss))]
print(G_loss[:][0])

plt.plot(epochs, G_loss[:][0], color='blue')
plt.plot(epochs, G_loss[:][1], color='green')
plt.plot(epochs, G_loss[:][2], color='orange')
plt.plot(epochs, D_loss, color='red')

plt.savefig('loss')
