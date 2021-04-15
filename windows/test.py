from PIL import Image
from tkinter import filedialog
import cv2


# ファイルダイアログからファイル選択
typ = [('','*')] 
dir = 'C:\\pg'
image_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
image = cv2.imread(image_path)

color_pallete = [0, 0, 0, 255, 255, 255]

with Image.fromarray(image, mode="P") as img:
    img.putpalette(color_palette)
    img.show()