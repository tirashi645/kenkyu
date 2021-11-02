import glob
import os
import shutil

li = ['train', 'test']

for type_data in li:
    cls_dirs = glob.glob('/media/koshiba/Data/sportConpetitive/'+type_data+'/**')

    for cls_dir in cls_dirs:
        dir_name = cls_dir.split('/')[-1]
        print(dir_name)
        files = glob.glob(cls_dir)
        print('/media/koshiba/Data/sportConpetitive/vgg/'+type_data+'/'+dir_name)
        for i, file in enumerate(files):
            if not os.path.exists('/media/koshiba/Data/sportConpetitive/vgg/'+type_data+'/'+dir_name):
                os.makedirs('/media/koshiba/Data/sportConpetitive/vgg/'+type_data+'/'+dir_name)
            
            shutil.copyfile(file, '/media/koshiba/Data/sportConpetitive/vgg/'+type_data+'/'+dir_name+'/'+dir_name+'_'+str(i)+'.jpg')