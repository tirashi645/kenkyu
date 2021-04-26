import numpy as np
import glob
import h5py
import cv2
from PIL import Image

from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator


inpath = '/media/koshiba/Data/pix2pix/input'
outpath = '/media/koshiba/Data/pix2pix/output'
'''
inpath = './input'
outpath = './output'
'''

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

orgs = []
masks = []
#seed = 123
masks_augment = np.array([])
org_augment = np.array([])

print('original img')
files = glob.glob(inpath+'/org/*.jpg')
for imgfile in files:
    print(imgfile)
    img = Image.open(imgfile)
    img = expand2square(img, (0, 0, 0))
    img = img.resize((256, 256))
    #img = load_img(imgfile, target_size=(256,256))
    imgarray = img_to_array(img)
    orgs.append(imgarray)
print(np.array(img).shape)
print('mask img')
files = glob.glob(inpath+'/mask/*.jpg')
for imgfile in files:
    print(imgfile)
    img = load_img(imgfile)
    #print(np.array(img).shape)
    img = expand2square(img, (0, 0, 0))
    img = img.resize((256, 256))
    #img = load_img(imgfile, target_size=(256,256))
    imgarray = img_to_array(img)
    masks.append(imgarray)
print(np.array(img).shape)

org_augment = org_augment.reshape([-1, 256, 256, 3])
masks_augment = masks_augment.reshape([-1, 256, 256, 3])

perm = np.random.permutation(len(orgs))
orgs = np.array(orgs)[perm]
masks = np.array(masks)[perm]
threshold = len(org_augment)//10*9
imgs = org_augment[:threshold]
gimgs = masks_augment[:threshold]
vimgs = org_augment[threshold:]
vgimgs = masks_augment[threshold:]
print('shapes')
print('org imgs  : ', imgs.shape)
print('mask imgs : ', gimgs.shape)
print('test org  : ', vimgs.shape)
print('test tset : ', vgimgs.shape)

outh5 = h5py.File(outpath+'/datasetimages.hdf5', 'w')
outh5.create_dataset('train_data_raw', data=imgs)
outh5.create_dataset('train_data_gen', data=gimgs)
outh5.create_dataset('val_data_raw', data=vimgs)
outh5.create_dataset('val_data_gen', data=vgimgs)
outh5.flush()
outh5.close()
print('finish')