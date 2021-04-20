import cv2
import matplotlib.pyplot as plt
import h5py
import numpy as np
from PIL import Image

f = h5py.File('./gitFile/kenkyu/windows/pix2pix/outputData.h5', 'r')

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


img = f['proc'][0]
print(f['proc'][0].shape)
 # アパーチャーサイズ 3, 5, or 7 など 1 より大きい奇数。数値が大きいほどぼかしが出る。
ksize=3
#中央値フィルタ
img_mask = cv2.medianBlur(img,ksize)
pil_img_mask = Image.fromarray((img_mask * 255).astype(np.uint8))
#plt.imshow(img_mask)
#plt.show()
pil_img_mask = pil_img_mask.resize((720, 1280), Image.LANCZOS)
pil_img_mask.save('./gitFile/kenkyu/windows/pix2pix/saveTest.jpg')
img_gray = pil2cv(pil_img_mask)
img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
img_sobel = cv2.Canny(img_gray, 100, 200)
cv2.imwrite('./gitFile/kenkyu/windows/pix2pix/saveEdge.jpg', img_sobel)