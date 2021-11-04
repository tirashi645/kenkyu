import os
import glob
import shutil

li1 = ['train', 'test']
cnt = 0
tmp = ""

classPath = glob.glob('/media/koshiba/Data/sportConpetitive/data2/**')
print(classPath)
for i, cls in enumerate(classPath):
    dataPath = glob.glob(cls + '/**/*.jpg', recursive=True)
    cls_name = cls.split('/')[-1]
    print(cls.split('/'))
    
    for j, data in enumerate(dataPath):
        if not os.path.exists('/media/koshiba/Data/sportConpetitive/vgg16/train/' + cls_name):
            os.makedirs('/media/koshiba/Data/sportConpetitive/vgg16/train/' + cls_name)
        if cls.split('/')[-1]=='ow':
            if data.split('/')[-2]!=tmp:
                cnt = 0
            if cnt%30==0:
                if not data.split('/')[-2]=='fig':
                    shutil.copyfile(data, '/media/koshiba/Data/sportConpetitive/vgg16/train/'+cls_name+'/'+cls_name+'_'+str(j)+'.jpg')
                else:
                    print(data)
            cnt += 1
        else:
            shutil.copyfile(data, '/media/koshiba/Data/sportConpetitive/vgg16/train/'+cls_name+'/'+cls_name+'_'+str(j)+'.jpg')