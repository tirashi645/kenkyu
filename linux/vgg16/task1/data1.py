import os
import glob
import shutil

li1 = ['train', 'test']

classPath = glob.glob('/media/koshiba/Data/sportConpetitive/data2/**/')
print(classPath)
for i, cls in enumerate(classPath):
    dataPath = glob.glob(cls + '/**/*.jpg', recursive=True)
    cls_name = cls.split('/')[-2]
    print(dataPath)
    
    for j, data in enumerate(dataPath):
        if not os.path.exists('/media/koshiba/Data/sportConpetitive/vgg_data/train/' + cls_name):
            os.makedirs('/media/koshiba/Data/sportConpetitive/vgg_data/train/' + cls_name)
        
        shutil.copyfile(data, '/media/koshiba/Data/sportConpetitive/vgg_data/train/' + cls_name+'/'+cls_name+'_'+str(i)+'.jpg')
        