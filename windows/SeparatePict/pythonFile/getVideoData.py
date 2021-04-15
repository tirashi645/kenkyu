def videoData(path):
    cap = cv2.VideoCapture(path)

    # 動画の設定を読み込み
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    rot = 0
    # 動画が横向きならば縦向きに回転させる
    if width>height:
        rot = 1
        tmp = width
        width = height
        height = tmp

    givenData = np.array([width, height, fps])

    print(givenData)    

    f = open(savePath + '/videoData.txt', 'wb')
    pickle.dump(givenData, f)

def getDirName(path):
    print(path)
    if 'inputVideo' in path:
        end = path.rfind('/')
        videoDir = path[path.rfind('/',0,end)+1:end]
    elif 'point_data' in path:
        videoDir = path[path.find('point_data')+len('point_data/'):]
        videoDir = videoDir[:videoDir.find('/')]
    else:
        videoDir = input()
    return videoDir
    
def getVideoName(path):
    videoDir = getDirName(path)
    tmp = path[path.find(videoDir)+len(videoDir)+1:]
    if 'inputVideo' in path:
        videoName = tmp[:-4]
    elif 'point_data' in path:
        videoName = tmp[:tmp.find('/')]
    else:
        videoName = input()
    return videoName

if __name__ == "__main__":
    from tkinter import filedialog
    import pickle
    import cv2
    import numpy as np
    import os

    typ = [('','*')] 
    if os.path.isdir('/media/koshiba/Data/opticalflow/point_data'):
        dir = '/media/koshiba/Data/opticalflow/point_data'
    else:
        dir = 'C:\\pg'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
    videoDir = path[:path.rfind('/')]
    dirName = videoDir[videoDir.rfind('/')+1:]
    videoName = path[path.rfind('/')+1:-4]
    dirPath = '/' + dirName + '/' + videoName
    savePath = '/media/koshiba/Data/opticalflow/point_data/' + dirName

    print(getVideoName(path))
    
    videoData(path)