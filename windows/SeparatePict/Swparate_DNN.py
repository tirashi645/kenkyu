<<<<<<< HEAD
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tkinter import filedialog
import os
import torchvision
from torchvision import transforms
from pythonFile import getVideoData


if __name__ == '__main__':
    
    # ファイルダイアログからファイル選択
    typ = [('','*')] 
    dir = 'C:\\pg'
    image_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 
    img = cv2.imread(image_path)
    img = img[...,::-1] #BGR->RGB
    h,w,_ = img.shape
    img = cv2.resize(img,(320,320))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(image_path)
    #dirName = 1_Trim
    #videoName = getVideoData.getVideoName(image_path)
    #savePath = 'D:/opticalflow/mask/' + dirName + '/' + videoName

    #if not os.path.exists(savePath):
    #    os.makedirs(savePath)

    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)['out'][0]
        output = output.argmax(0)
        mask = output.byte().cpu().numpy()
        mask = cv2.resize(mask,(w,h))
        img = cv2.resize(img,(w,h))
        plt.gray()
        plt.figure(figsize=(20,20))
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(mask)
=======
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tkinter import filedialog
import os
import torchvision
from torchvision import transforms
from pythonFile import getVideoData


if __name__ == '__main__':
    
    # ファイルダイアログからファイル選択
    typ = [('','*')] 
    dir = 'C:\\pg'
    image_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 
    img = cv2.imread(image_path)
    img = img[...,::-1] #BGR->RGB
    h,w,_ = img.shape
    img = cv2.resize(img,(320,320))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(image_path)
    #dirName = 1_Trim
    #videoName = getVideoData.getVideoName(image_path)
    #savePath = 'D:/opticalflow/mask/' + dirName + '/' + videoName

    #if not os.path.exists(savePath):
    #    os.makedirs(savePath)

    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)['out'][0]
        output = output.argmax(0)
        mask = output.byte().cpu().numpy()
        mask = cv2.resize(mask,(w,h))
        img = cv2.resize(img,(w,h))
        plt.gray()
        plt.figure(figsize=(20,20))
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(mask)
>>>>>>> bdd2750e416964698f1ddbe1736dcfb1853f2963
        cv2.imwrite('D:/opticalflow/mask/test.jpg', mask)