import glob
import shutil
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
import random
import os, cv2, random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import ticker
from PIL import Image
    
li = ['train', 'test']

def rand_ints(a, b, k):
  ns = []
  while len(ns) < k:
    n = random.randint(a, b)
    if not n in ns:
      ns.append(n)
  return ns

TRAIN_DIR = "/media/koshiba/Data/sportConpetitive/refree/train/"
TEST_DIR = "/media/koshiba/Data/sportConpetitive/refree/test/"
OUTPUT_DIR = "/media/koshiba/Data/sportConpetitive/refree/output/"

train_refree = [TRAIN_DIR+'ippon/' + i for i in os.listdir(TRAIN_DIR+'ippon/')]
train_player = [TRAIN_DIR+'wazaari/' + i for i in os.listdir(TRAIN_DIR+'wazaari/')]
train_ow = [TRAIN_DIR+'normal/' + i for i in os.listdir(TRAIN_DIR+'normal/')]

test_refree = [TEST_DIR+'ippon/' + i for i in os.listdir(TEST_DIR+'ippon/')]
test_player = [TEST_DIR+'wazaari/' + i for i in os.listdir(TEST_DIR+'wazaari/')]
test_ow = [TEST_DIR+'normal/' + i for i in os.listdir(TEST_DIR+'normal/')]

#test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)]
train_images = train_refree + train_player + train_ow
test_images = test_refree + test_player + test_ow

params = dict(featurewise_center=True,
            featurewise_std_normalization=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False)
image_datagen = image.ImageDataGenerator(**params)

orgs = []
org_augment = np.array([])

for file_num, dataPath in enumerate(train_images):
    cv2.imread(dataPath)
    num_list = rand_ints(0, 17, 4)
    img_name = train_images[file_num].split('/')[-1][:-4]

    print(train_images[file_num])
    img = Image.open(train_images[file_num])
    img = img.resize((256, 256))
    #img = load_img(imgfile, target_size=(256,256))
    imgarray = img_to_array(img)
    orgs.append(imgarray)

    seed = np.random.randint(1, 1000)
    img2 = orgs[-1]
    for i, data in enumerate(image_datagen.flow(img2[np.newaxis, :, :, :], y=None, batch_size=1, shuffle=False, seed=seed)):
        data = cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY)
        cv2.imwrite(dataPath[:-4] + str(i) + '.jpg')
        if i == 4:
            break