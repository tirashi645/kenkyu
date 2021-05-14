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
upper_orgs = []
lower_orgs = []
upper_masks = []
lower_masks = []
#seed = 123
upper_masks_augment = np.array([])
upper_org_augment = np.array([])
lower_masks_augment = np.array([])
lower_org_augment = np.array([])

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

def process(img):
    img = expand2square(img, (0, 0, 0))
    img = img.resize((256, 256))
    #img = load_img(imgfile, target_size=(256,256))
    return img_to_array(img)

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


print('original img')
files_org = glob.glob(inpath+'/org/*.jpg')
files_mask = glob.glob(inpath+'/mask/*.jpg')
for file_num, data in enumerate(files_org):
    num_list = rand_ints(0, 17, 4)
    img_name = files_org[file_num].split('/')[-1][:-4]

    print(files_org[file_num])
    img = Image.open(files_org[file_num])
    width, height = img.size
    upper = img.crop((0, 0, width, height/2))
    lower = img.crop((0, height/2, width, height))
    upper_imgarray_org = process(upper)
    lower_imgarray_org = process(lower)
    upper_orgs.append(upper_imgarray_org)
    lower_orgs.append(lower_imgarray_org)

    img = load_img(files_mask[file_num])
    upper = img.crop((0, 0, width, height/2))
    lower = img.crop((0, height/2, width, height))
    upper_imgarray_mask = process(upper)
    lower_imgarray_mask = process(lower)
    upper_masks.append(upper_imgarray_mask)
    lower_masks.append(lower_imgarray_mask)

    
    seed = np.random.randint(1, 1000)
    img1 = upper_masks[-1]
    img2 = lower_masks[-1]
    img3 = upper_orgs[-1]
    img4 = lower_orgs[-1]
    for i, data in enumerate(mask_datagen.flow(img1[np.newaxis, :, :, :], y=None, batch_size=1, shuffle=False, seed=seed)):
        print(type(data), data.shape)
        data = cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY)
        upper_masks_augment = np.append(upper_masks_augment, data)
        if i == 4:
            break
    for i, data in enumerate(image_datagen.flow(img2[np.newaxis, :, :, :], y=None, batch_size=1, shuffle=False, seed=seed)):
        data = cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY)
        lower_masks_augment = np.append(lower_masks_augment, data)
        if i == 4:
            break
    for i, data in enumerate(mask_datagen.flow(img3[np.newaxis, :, :, :], y=None, batch_size=1, shuffle=False, seed=seed)):
        print(type(data), data.shape)
        data = cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY)
        upper_org_augment = np.append(upper_org_augment, data)
        if i == 4:
            break
    for i, data in enumerate(image_datagen.flow(img4[np.newaxis, :, :, :], y=None, batch_size=1, shuffle=False, seed=seed)):
        data = cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY)
        lower_org_augment = np.append(lower_org_augment, data)
        if i == 4:
            break

    for i in num_list:
        org_img = '/media/koshiba/Data/pix2pix/input/synthetic/' + img_name + '_' + str(i) + '.jpg'
        print(org_img)
        img = Image.open(org_img)
        img = expand2square(img, (0, 0, 0))
        img = img.resize((256, 256))
        #img = load_img(imgfile, target_size=(256,256))
        imgarray = img_to_array(img)
        upper_orgs.append(upper_imgarray_org)
        upper_masks.append(upper_imgarray_mask)
        lower_orgs.append(lower_imgarray_org)
        lower_masks.append(upper_imgarray_mask)
        img1 = upper_masks[-1]
        img2 = lower_masks[-1]
        img3 = upper_orgs[-1]
        img4 = lower_orgs[-1]
        for i, data in enumerate(mask_datagen.flow(img1[np.newaxis, :, :, :], y=None, batch_size=1, shuffle=False, seed=seed)):
            print(type(data), data.shape)
            data = cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY)
            upper_masks_augment = np.append(upper_masks_augment, data)
            if i == 4:
                break
        for i, data in enumerate(image_datagen.flow(img2[np.newaxis, :, :, :], y=None, batch_size=1, shuffle=False, seed=seed)):
            data = cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY)
            lower_masks_augment = np.append(lower_masks_augment, data)
            if i == 4:
                break
        for i, data in enumerate(mask_datagen.flow(img3[np.newaxis, :, :, :], y=None, batch_size=1, shuffle=False, seed=seed)):
            print(type(data), data.shape)
            data = cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY)
            upper_org_augment = np.append(upper_org_augment, data)
            if i == 4:
                break
        for i, data in enumerate(image_datagen.flow(img4[np.newaxis, :, :, :], y=None, batch_size=1, shuffle=False, seed=seed)):
            data = cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY)
            lower_org_augment = np.append(lower_org_augment, data)
            if i == 4:
                break

upper_org_augment = upper_org_augment.reshape([-1, 256, 256, 1])
upper_masks_augment = upper_masks_augment.reshape([-1, 256, 256, 1])
lower_org_augment = lower_org_augment.reshape([-1, 256, 256, 1])
lower_masks_augment = lower_masks_augment.reshape([-1, 256, 256, 1])


# upper data sets
perm = np.random.permutation(len(upper_orgs))
orgs = np.array(upper_orgs)[perm]
masks = np.array(upper_masks)[perm]
threshold = len(upper_org_augment)//10*9
imgs = upper_org_augment[:threshold]
gimgs = upper_masks_augment[:threshold]
vimgs = upper_org_augment[threshold:]
vgimgs = upper_masks_augment[threshold:]
print('upper shapes')
print('org imgs  : ', imgs.shape)
print('mask imgs : ', gimgs.shape)
print('test org  : ', vimgs.shape)
print('test tset : ', vgimgs.shape)

outh5 = h5py.File(outpath+'/datasetimages_upper.hdf5', 'w')
outh5.create_dataset('train_data_raw', data=imgs)
outh5.create_dataset('train_data_gen', data=gimgs)
outh5.create_dataset('val_data_raw', data=vimgs)
outh5.create_dataset('val_data_gen', data=vgimgs)
outh5.flush()
outh5.close()
print('finish')

# lower data sets
orgs = np.array(lower_orgs)[perm]
masks = np.array(lower_masks)[perm]
threshold = len(lower_org_augment)//10*9
imgs = lower_org_augment[:threshold]
gimgs = lower_masks_augment[:threshold]
vimgs = lower_org_augment[threshold:]
vgimgs = lower_masks_augment[threshold:]
print('lower shapes')
print('org imgs  : ', imgs.shape)
print('mask imgs : ', gimgs.shape)
print('test org  : ', vimgs.shape)
print('test tset : ', vgimgs.shape)

outh5 = h5py.File(outpath+'/datasetimages_lower.hdf5', 'w')
outh5.create_dataset('train_data_raw', data=imgs)
outh5.create_dataset('train_data_gen', data=gimgs)
outh5.create_dataset('val_data_raw', data=vimgs)
outh5.create_dataset('val_data_gen', data=vgimgs)
outh5.flush()
outh5.close()
print('finish')