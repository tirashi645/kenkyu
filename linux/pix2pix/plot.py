import pickle
import matplotlib.pyplot as plt

outputpath = '/media/koshiba/Data/pix2pix/output'

with open(outputpath + '/gloss.pkl', 'rb') as f:
    G_loss = pickle.load(f)
with open(outputpath + '/dloss.pkl', 'rb') as f:
    D_logloss = pickle.load(f)

epochs = [i+1 for i in range(len(D_logloss))]
G_tot = [i[0] for i in G_loss]
l1 = [i[1] for i in G_loss]
G_logloss = [i[2] for i in G_loss]

plt.figure()
plt.plot(epochs, G_tot, color='blue')
plt.savefig(outputpath + '/G_tot.jpg')

plt.figure()
plt.plot(epochs, l1, color='green')
plt.savefig(outputpath + '/l1.jpg')

plt.figure()
plt.plot(epochs, G_logloss, color='orange')
plt.savefig(outputpath + '/G_loglossloss.jpg')

plt.figure()
plt.plot(epochs, D_logloss, color='red')
plt.savefig(outputpath + '/D_logloss.jpg')
