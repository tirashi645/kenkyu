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
from keras.preprocessing.image import load_img, img_to_array
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

def normalization(X):
    return X / 127.5 - 1

def inverse_normalization(X):
    return (X + 1.) / 2.

def to3d(X):
    if X.shape[-1]==3: return X
    b = X.transpose(3,1,2,0)
    c = np.array([b[0],b[0],b[0]])
    return c.transpose(3,1,2,0)

def plot_generated_batch(X_raw, generator_model, batch_size, b_id, num):
    X_gen = generator_model.predict(X_raw)
    X_raw = inverse_normalization(X_raw)
    X_gen = inverse_normalization(X_gen)

    for i in range(len(X_gen)):
        if i>=num:
            break
        print(X_gen[i].shape)
        Xg = X_gen[i]
        X_raw[i, :, 0], X_raw[i, :, 2] = X_raw[i, :, 2], X_raw[i, :, 0].copy()
        cv2.imwrite(outputpath + "/proc_tmp/gen" + str(b_id) + '_' +str(i)+".jpg", np.array(Xg) * 255)
        cv2.imwrite(outputpath + "/proc_tmp/raw" + str(b_id) + '_' +str(i)+".jpg", np.array(X_raw[i]) * 255)

def proc_generator_batch(X_raw, generator_model, batch_size, b_id, num, img_size):
    X_gen = generator_model.predict(X_raw)
    X_gen = inverse_normalization(X_gen)
    X_gen = gen_resize(X_gen, img_size)

    if img_size[0]==img_size[1]:
        return X_gen[:min(batch_size, num)]
    if img_size[0]<img_size[1]:
        padding_num = (img_size[1] - img_size[0]) // 2
        return X_gen[:min(batch_size, num), padding_num:padding_num+1, :]
    else:
        padding_num = (img_size[0] - img_size[1]) // 2 
        return X_gen[:min(batch_size, num), :, padding_num:padding_num+1, :]

def gen_resize(x, img_size):
    X_gen = np.array([])
    for data in x:
        X_gen = np.append(X_gen, cv2.resize(data, (max(img_size[0], img_size[1]), max(img_size[0], img_size[1]))))
    return X_gen.reshape([-1, max(img_size[0], img_size[1]), max(img_size[0], img_size[1]), 3])

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

    proc_file = glob.glob(inputpath + '/proc_tmp/*.jpg')
    img_list = np.array([])     # generatorの入力画像
    org_list = np.array([])     # オリジナルの画像
    gen_list = np.array([])     # generatorの出力画像
    num = 0                     # 入力画像の枚数
    name_list = []
    flag = True
    for img_file in proc_file:
        name_list.append(img_file[img_file.rfind('/')+1:img_file.rfind('.')])   # 画像の名前を取得
        num += 1
        img_name = img_file.split('/')[-1]
        org_img = Image.open(img_file)            # PILで画像読み込み
        cv2_img = pil2cv(org_img)
        org_list = np.append(org_list, cv2_img)
        if flag:
            width, height = org_img.size
            img_size = [height, width]
            flag = False
        img = expand2square(org_img, (0, 0, 0))
        img = img.resize((256, 256))
        img = img_to_array(img)
        img_list = np.append(img_list, img)
    img_list = normalization(img_list)
    if num<batch_size:
        for _ in range(batch_size - num):
            img_list = np.append(img_list, img)


    img_list = img_list.reshape([-1, 256, 256, 3])
    org_list = org_list.reshape([-1, height, width, 3])
    img_procImageIter = np.array([img_list[i:i+batch_size] for i in range(0, img_list.shape[0], batch_size)])
    print(img_procImageIter.shape)
    for index, proc_batch in enumerate(img_procImageIter):
        print(proc_batch.shape)
        #plot_generated_batch(proc_batch, generator_model, batch_size, b_id, num)
        gen_list = np.append(gen_list, proc_generator_batch(proc_batch, generator_model, batch_size, b_id, num, img_size))
        b_id += 1
    #gen_list = np.reshape([-1, height, width, 3])
    print(gen_list.shape)
    for index in range(min(num, len(gen_list))):
        print(org_list[index].shape)
        cv2.imwrite(outputpath + "/proc_tmp/raw_" + name_list[index] +".jpg", org_list[index])
        cv2.imwrite(outputpath + "/proc_tmp/gen_" + name_list[index] +".jpg", gen_list[index] * 255)

if __name__ == '__main__':
    #train()
    proc()
