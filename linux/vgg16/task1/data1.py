import os
import glob
import shutil

li1 = ['train', 'test']

classPath = glob.glob('/media/koshiba/Data/sportConpetitive/data2/**/')
print(classPath)
for i, cls in enumerate(classPath):
    dataPath = glob.glob(cls + '/**/*.jpg', recursive=True)
    cls_name = cls.split('/')[-2]
    print(cls_name)
    
    for j, data in enumerate(dataPath):
        if not os.path.exists('/media/koshiba/Data/sportConpetitive/vgg_data/train/' + cls_name):
            os.makedirs('/media/koshiba/Data/sportConpetitive/vgg_data/train/' + cls_name)
        if not data.split('/')[-2]=='fig':
            shutil.copyfile(data, '/media/koshiba/Data/sportConpetitive/vgg_data/train/'+cls_name+'/'+cls_name+'_'+str(j)+'.jpg')
        else:
            print(data)
        