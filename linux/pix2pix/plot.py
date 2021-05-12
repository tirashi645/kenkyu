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

fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()

fig1.plot(epochs, G_tot, color='blue')
fig2.plot(epochs, l1, color='green')
fig3.plot(epochs, G_logloss, color='orange')
fig4.plot(epochs, D_logloss, color='red')

fig1.savefig(outputpath + '/G_tot.jpg')
fig2.savefig(outputpath + '/l1.jpg')
fig3.savefig(outputpath + '/G_logloss.jpg')
fig4.savefig(outputpath + '/D_logloss.jpg')
