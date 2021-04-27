import numpy as np

import h5py
import time
import cv2
from PIL import Image
import glob

import matplotlib.pylab as plt

import keras.backend as K
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD

from keras.models import Model, load_model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
from keras.callbacks import TensorBoard

model_dir = '/media/koshiba/Data/pix2pix/model'
log_dir = './tflog'
datasetpath = '/media/koshiba/Data/pix2pix/output/datasetimages.hdf5'
outputpath = '/media/koshiba/Data/pix2pix/output'
inputpath = '/media/koshiba/Data/pix2pix/input'
'''
procinputpath = '/media/koshiba/Data/pix2pix/proc/input'
procoutputpath = '/media/koshiba/Data/pix2pix/proc/output'

model_dir = './model'
log_dir = './tflog'
datasetpath = './output/datasetimages.hdf5'
outputpath = './output'
'''

patch_size = 32
batch_size = 12
epoch = 10

def inverse_normalization(X):
    return (X + 1.) / 2.

def to3d(X):
    if X.shape[-1]==3: return X
    b = X.transpose(3,1,2,0)
    c = np.array([b[0],b[0],b[0]])
    return c.transpose(3,1,2,0)

def plot_generated_batch(X_raw, generator_model, batch_size, b_id):
    X_gen = generator_model.predict(X_raw)
    X_raw = inverse_normalization(X_raw)
    X_gen = inverse_normalization(X_gen)

    '''
    with h5py.File(outputpath + '/outputData.h5', 'w') as f:
        f.create_dataset('raw', data=X_raw)
        f.create_dataset('gen', data=X_gen)
    '''
    for i in range(len(X_gen)):
        print(X_gen[i].shape)
        '''
        Xs = X_raw[i]
        Xg = X_gen[i]
        Xs = np.concatenate(Xs, axis=1)
        Xg = np.concatenate(Xg, axis=1)
        XX = np.concatenate((Xs,Xg), axis=0)

        plt.imshow(XX)
        plt.axis('off')
        plt.savefig(outputpath + "/proc/batch" + str(b_id) + '_' +str(i)+".png")
        plt.clf()
        plt.close()
        '''
        Xg = pil2cv(X_gen[i])
        cv2.imwrite(outputpath + "/proc/gen" + str(b_id) + '_' +str(i)+".jpg", np.array(Xg)*255)
        cv2.imwrite(outputpath + "/proc/raw" + str(b_id) + '_' +str(i)+".jpg", np.array(X_raw[i]))

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def pil2cv(image):
  ''' PIL型 -> OpenCV型 '''
  new_image = np.array(image, dtype=np.uint8)
  if new_image.ndim == 2:  # モノクロ
      pass
  elif new_image.shape[2] == 3:  # カラー
      new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
  elif new_image.shape[2] == 4:  # 透過
      new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
  return new_image

def proc():
    b_id = 0
    generator_model = load_model(model_dir + '/genenrator.h5')
    generator_model.load_weights(model_dir + '/genenrator_weights.h5')

    proc_file = glob.glob(inputpath + '/proc/*.jpg')
    img_list = np.array([])
    i = 0
    for img_file in proc_file:
        i += 1
        img_name = img_file.split('/')[-1]
        img = Image.open(img_file)
        img = expand2square(img, (0, 0, 0))
        img = img.resize((256, 256))
        img = pil2cv(img)
        img_list = np.append(img_list, img)
    if i<batch_size:
        for _ in range(batch_size - i):
            img_list = np.append(img_list, img)

    img_list = img_list.reshape([-1, 256, 256, 3])
    img_procImageIter = np.array([img_list[i:i+batch_size] for i in range(0, img_list.shape[0], batch_size)])
    print(img_procImageIter.shape)
    for proc_batch in img_procImageIter:
        print(proc_batch.shape)
        plot_generated_batch(proc_batch, generator_model, batch_size, b_id)
        b_id += 1


if __name__ == '__main__':
    #train()
    proc()
