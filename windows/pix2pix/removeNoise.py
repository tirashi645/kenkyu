import cv2
import matplotlib.pyplot as plt
import h5py
import numpy as np
from PIL import Image

f = h5py.File('./gitFile/kenkyu/windows/pix2pix/outputData.h5', 'r')


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