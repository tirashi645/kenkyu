import numpy as np
import glob
from PIL import Image

model_dir = '/media/koshiba/Data/pix2pix/model'
log_dir = './tflog'
datasetpath = '/media/koshiba/Data/pix2pix/output/datasetimages.hdf5'
outputpath = '/media/koshiba/Data/pix2pix/output'
inputpath = '/media/koshiba/Data/pix2pix/input'

proc_file = glob.glob(inputpath + '/proc_tmp/*.jpg')
for img in proc_file:
    print(img.split('/')[-2], img.split('/')[-1])