import cv2
from tkinter import filedialog

# ファイルダイアログからファイル選択
typ = [('','*')] 
dir = 'C:\\pg'
image_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)

cap = cv2.VideoCapture(image_path)
wait_secs = int(1000 / cap.get(cv2.CAP_PROP_FPS))

model = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask = model.apply(frame)

    cv2.imshow("Mask", mask)
    cv2.waitKey(wait_secs)

cap.release()
cv2.destroyAllWindows()