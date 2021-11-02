import glob
import os

cls_dirs = glob.glob('/media/koshiba/Data/sportConpetitive/train/**')

for cls_dir in cls_dirs:
    dir_name = cls_dir.split('/')[-1]
    print(dir_name)