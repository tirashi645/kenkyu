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
=======
from PIL import Image
from tkinter import filedialog


# ファイルダイアログからファイル選択
typ = [('','*')] 
dir = 'C:\\pg'
image_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)

image = Image.open(image_path)
print(image.mode)
>>>>>>> bdd2750e416964698f1ddbe1736dcfb1853f2963
