from PIL import Image

INPUT_DIR = ""
OUTPUT_DIR = ""

img = Image.open(INPUT_DIR)
img_flr = img.transpose(Image.FLIP_LEFT_RIGHT)
img_flr.save(OUTPUT_DIR)
img_rotate = img.rotate(15)
img_rotate.save(OUTPUT_DIR)