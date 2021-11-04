import os
import glob

classPath = glob.glob('/media/koshiba/Data/sportConpetitive/data2/')

for cls in classPath:
    dataPath = glob.glob(cls + '/**/**/*')
    print(dataPath[0])