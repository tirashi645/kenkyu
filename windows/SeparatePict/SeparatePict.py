import cv2
from tkinter import filedialog
import os
from pythonFile import getVideoData

if __name__ == '__main__':
    cnt = 0

    # ファイルダイアログからファイル選択
    typ = [('','*')] 
    dir = 'C:\\pg'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 

    dirName = getVideoData.getDirName(path)
    videoName = getVideoData.getVideoName(path)
    savePath = 'D:/opticalflow/pict/' + dirName + '/' + videoName

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # 読み込む動画の設定
    videoName = path[path.rfind('/')+1:]
    cap = cv2.VideoCapture(path)
    
    while(True):
        # 最初のフレームを読み込む
        ret, frame = cap.read()
        if ret==False:
            break
        if cnt%10 == 0:
            cv2.imwrite(savePath + '/' + str(cnt) + '.jpg', frame)
        cnt += 1