<<<<<<< HEAD
from PIL import Image
from tkinter import filedialog


# ファイルダイアログからファイル選択
typ = [('','*')] 
dir = 'C:\\pg'
image_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)

image = Image.open(image_path)
print(image.mode)

print(image_path.split('/')[0])