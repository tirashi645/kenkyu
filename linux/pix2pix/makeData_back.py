import numpy as np
import glob
import h5py
import cv2
from PIL import Image
import random

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

def rand_ints(a, b, k):
  ns = []
  while len(ns) < k:
    n = random.randint(a, b)
    if not n in ns:
      ns.append(n)
  return ns

# we create two instances with the same arguments
data_gen_args1 = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.1,
                     horizontal_flip=False,
                     channel_shift_range=30)
data_gen_args2 = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.1,
                     horizontal_flip=False)
image_datagen = ImageDataGenerator(**data_gen_args1)
mask_datagen = ImageDataGenerator(**data_gen_args2)

orgs = []
masks = []
#seed = 123
masks_augment = np.array([])
org_augment = np.array([])

print('original img')
files_org = glob.glob(inpath+'/org/*.jpg')
files_mask = glob.glob(inpath+'/mask/*.jpg')
for file_num, data in enumerate(files_org):
    num_list = rand_ints(0, 17, 4)
    img_name = files_org[file_num].split('/')[-1][:-4]
    img_dir = files_org[file_num].split('/')[-2]

    print(files_org[file_num])
    img = Image.open(files_org[file_num])
    img = expand2square(img, (0, 0, 0))
    img = img.resize((256, 256))
    #img = load_img(imgfile, target_size=(256,256))
    imgarray = img_to_array(img)
    orgs.append(imgarray)

    img = load_img(files_mask[file_num])
    #print(np.array(img).shape)
    img = expand2square(img, (0, 0, 0))
    img = img.resize((256, 256))
    #img = load_img(imgfile, target_size=(256,256))
    imgarray_mask = img_to_array(img)
    masks.append(imgarray_mask)

    seed = np.random.randint(1, 1000)
    img1 = masks[-1]
    img2 = orgs[-1]
    for i, data in enumerate(mask_datagen.flow(img1[np.newaxis, :, :, :], y=None, batch_size=1, shuffle=False, seed=seed)):
        print(type(data), data.shape)
        data = cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY)
        masks_augment = np.append(masks_augment, data)
        if i == 4:
            break
    for i, data in enumerate(image_datagen.flow(img2[np.newaxis, :, :, :], y=None, batch_size=1, shuffle=False, seed=seed)):
        data = cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY)
        org_augment = np.append(org_augment, data)
        if i == 4:
            break

    for i in num_list:
        org_img = '/media/koshiba/Data/pix2pix/input/synthentic/' + img_dir + '/' + img_name + '_' + str(i) + '.jpg'
        print(org_img)
        img = Image.open(org_img)
        img = expand2square(img, (0, 0, 0))
        img = img.resize((256, 256))
        #img = load_img(imgfile, target_size=(256,256))
        imgarray = img_to_array(img)
        orgs.append(imgarray)
        masks.append(imgarray_mask)
        img1 = masks[-1]
        img2 = orgs[-1]
        for i, data in enumerate(mask_datagen.flow(img1[np.newaxis, :, :, :], y=None, batch_size=1, shuffle=False, seed=seed)):
            print(type(data), data.shape)
            data = cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY)
            masks_augment = np.append(masks_augment, data)
            if i == 4:
                break
        for i, data in enumerate(image_datagen.flow(img2[np.newaxis, :, :, :], y=None, batch_size=1, shuffle=False, seed=seed)):
            data = cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY)
            org_augment = np.append(org_augment, data)
            if i == 4:
                break


'''
for index in range(len(masks)):
    seed = np.random.randint(1, 1000)
    img1 = masks[index]
    img2 = orgs[index]
    for i, data in enumerate(mask_datagen.flow(img1[np.newaxis, :, :, :], y=None, batch_size=1, shuffle=False, seed=seed)):
        print(type(data), data.shape)
        data = cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY)
        masks_augment = np.append(masks_augment, data)
        if i == 4:
            break
    for i, data in enumerate(image_datagen.flow(img2[np.newaxis, :, :, :], y=None, batch_size=1, shuffle=False, seed=seed)):
        data = cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY)
        org_augment = np.append(org_augment, data)
        if i == 4:
            break
'''

org_augment = org_augment.reshape([-1, 256, 256, 1])
masks_augment = masks_augment.reshape([-1, 256, 256, 1])


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