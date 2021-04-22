from tensorflow.keras.preprocessing.image import ImageDataGenerator
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def normalization(X):
    return X / 127.5 - 1

def load_data(datasetpath):
    with h5py.File(datasetpath, "r") as hf:
        X_full_train = hf["train_data_raw"][:].astype(np.float32)
        X_full_train = normalization(X_full_train)
        X_sketch_train = hf["train_data_gen"][:].astype(np.float32)
        X_sketch_train = normalization(X_sketch_train)
        X_full_val = hf["val_data_raw"][:].astype(np.float32)
        X_full_val = normalization(X_full_val)
        X_sketch_val = hf["val_data_gen"][:].astype(np.float32)
        X_sketch_val = normalization(X_sketch_val)
        return X_full_train, X_sketch_train, X_full_val, X_sketch_val

datasetpath = '/media/koshiba/Data/pix2pix/output/datasetimages.hdf5'
rawImage, procImage, rawImage_val, procImage_val = load_data(datasetpath)

inpath = './input'
outpath = './output'

# we create two instances with the same arguments
data_gen_args1 = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=0.2,
                     channel_shift_range=1)
data_gen_args2 = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=0.2)
image_datagen = ImageDataGenerator(**data_gen_args1)
mask_datagen = ImageDataGenerator(**data_gen_args2)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 123
masks_augment = np.array([])
org_augment = np.array([])
i = 0

'''
for data in image_datagen.flow_from_directory('/home/koshiba/anaconda3/envs/gitFolder/kenkyu/linux/pix2pix/augment/org',class_mode=None, target_size=(256, 256),seed=seed):
        org_augment = np.append(org_augment, data)
        i += 1
        if i == 4:
            break

validation_generator = mask_datagen.flow_from_directory(
    '/home/koshiba/anaconda3/envs/gitFolder/kenkyu/linux/pix2pix/augment/mask',
    target_size=(256, 256),
    batch_size=1,
    seed=seed)

for img in rawImage:
    img = img[np.newaxis, :, :, :]
    for i, data in enumerate(image_datagen.flow(img, y=None, batch_size=1, shuffle=False, seed=seed)):
        org_augment = np.append(org_augment, data)
        if i == 4:
            break
org_augment = org_augment.reshape([-1, 256, 256, 3])
for i, procImage in enumerate(mask_datagen.flow(procImage, y=None, batch_size=1, shuffle=False, seed=seed, save_to_dir='/home/koshiba/anaconda3/envs/gitFolder/kenkyu/linux/pix2pix/augment/mask')):
    if i == 10:
        break

rawImage_val = image_datagen.flow(rawImage_val, y=None, batch_size=1, shuffle=False, seed=seed)

procImage_val = mask_datagen.flow(procImage_val, y=None, batch_size=1, shuffle=False, seed=seed)

'''

pritn(rawImage.shape, procImage.Shape)