from PIL import Image
import requests
import glob
import cv2
#file_path = '/content/drive/My Drive/Colab Notebooks/CenterNet/images/' + fileName + '.mp4' #動画ファイルのパス
save_path = '/media/koshiba/Data/transformers/inputdata/'

# imageフォルダ内のファイルパスをすべて取得する
image_list = glob.glob('/media/koshiba/Data/simple-HRNet/inputData/**/*.mp4')
print(image_list)

for image_path in image_list:
    cap = cv2.VideoCapture(image_path)
    ret, frame = cap.read()
    cv2.imwrite(save_path + image_path.split('/')[-1] + '.jpg')