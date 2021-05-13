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
files = glob.glob(inpath+'/org/*.jpg')
for imgfile in files:
    print(imgfile)
    img = Image.open(imgfile)
    width, height = img.size
    print(type(height), height, type(width), width)
    upper = img.crop((0, 0, width, height/2))
    lower = img.crop((0, height/2, width, height))
    upper_orgs.append(process(upper))
    lower_orgs.append(process(lower))
print(np.array(img).shape)
print(height, width)
print(upper.size)
print('mask img')
files = glob.glob(inpath+'/mask/*.jpg')
for imgfile in files:
    print(imgfile)
    img = load_img(imgfile)
    upper = img[:height/2]
    lower = img[height/2:]
    upper_masks.append(process(upper))
    lower_masks.append(process(lower))
print(np.array(img).shape)

# augment data
for index in range(len(upper_masks)):
    seed = np.random.randint(1, 1000)
    img1 = upper_masks[index]
    img2 = lower_masks[index]
    img3 = upper_orgs[index]
    img4 = lower_orgs[index]
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