if __name__ == "__main__":
    from tkinter import filedialog
    import pickle
    import cv2

    typ = [('','*')] 
    dir = 'C:\\pg'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
    videoDir = path[:path.rfind('/')]
    dirName = videoDir[videoDir.rfind('/')+1:]
    videoName = path[path.rfind('/')+1:-4]
    dirPath = '/' + dirName + '/' + videoName
    savePath = 'D:/opticalflow/point_data/' + dirName + '/' + videoName
    
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

    givenData = [str(width), str(height), str(fps)]

    print(givenData)

    f = open(savePath + '/videoData_' + videoName + '.txt', 'wb')
    pickle.dump(givenData, f)