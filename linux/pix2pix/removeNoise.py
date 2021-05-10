import cv2
import matplotlib.pyplot as plt
import h5py
import numpy as np
from PIL import Image

#f = h5py.File('E:/data/pix2pix/output/outputData.h5', 'r')

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def todo(image):
    # アパーチャーサイズ 3, 5, or 7 など 1 より大きい奇数。数値が大きいほどぼかしが出る。
    ksize=3
    #中央値フィルタ
    print(type(image), image.shape)
    #image = np.where(image.sum(axis=2) > 0, 255, 0).astype(np.uint8)
    image = image.astype(np.uint8)
    img_mask = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    img_mask = closing = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)
    #img_mask = cv2.medianBlur(image, ksize)
    #img_mask = np.where(img_mask.sum(axis=2) > 0, 255, 0)

    '''
    pil_img_mask = Image.fromarray((img_mask * 255).astype(np.uint8))
    #plt.imshow(img_mask)
    #plt.show()
    pil_img_mask = pil_img_mask.resize((720, 1280), Image.LANCZOS)
    '''
    cv2.imwrite('E:/data/pix2pix/output/saveTest.jpg', img_mask)
    #img_gray = cv2.cvtColor(img_mask, cv2.COLOR_RGB2GRAY)
    #img_sobel = cv2.Canny(img_gray, 100, 200)
    #cv2.imwrite('E:/data/pix2pix/output/saveEdge.jpg', img_sobel)
    kernel = np.ones((20,20),np.uint8)
    img_dilation =  cv2.dilate(img_mask, kernel, iterations=1)
    #cv2.imwrite('/media/koshiba/Data/pix2pix/output/proc_point', img_dilation)

    return img_dilation

if __name__ == '__main__':
    pass