def doGet(path, savepath, videoPath, ws=15):
    import cv2
    import pickle
    import numpy as np
    from pythonFile import getVideoData

    # 動画読み込み
    cap = cv2.VideoCapture(videoPath)
    videoName = getVideoData.getVideoName(path)

    # 座標ファイル読み込み
    f = open(path, 'rb')
    zahyou = pickle.load(f)

    # 動画読み込み設定
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    print(savepath + '/' + videoName)
    writer = cv2.VideoWriter(savepath + '/' + videoName + '_' + ws + '.avi', fourcc, fps, (width, height))

    max_len = max(map(len, zahyou))
    c = [[255, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255]]    # 特徴点の色

    for frameNum in range(max_len):
        ret, first_frame = cap.read()
        flow_layer = np.zeros_like(first_frame)
        for point in range(len(zahyou)):
            if frameNum<len(zahyou[point]):
                flow_layer = cv2.circle(
                                                flow_layer,
                                                (int(zahyou[point][frameNum][0]), int(zahyou[point][frameNum][1])),
                                                2,
                                                color = c[3],
                                                thickness=3
                                            )
        frame = cv2.add(first_frame, flow_layer)
        writer.write(frame)

    writer.release()
    cap.release()
    print('finish')

if __name__ == "__main__":
    from tkinter import filedialog
    import glob
    import os
    from pythonFile import getVideoData, make_dirs

    typ = [('','*')] 
    if os.path.isdir('/media/koshiba/Data/opticalflow/point_data'):
        dir = '/media/koshiba/Data/opticalflow/point_data'
    else:
        dir = 'C:\\pg'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
    videoName = getVideoData.getVideoName(path)
    dirName = getVideoData.getDirName(path)
    fileName = path[path.rfind('/')+1:]
    videoDir = path[:path.rfind('/')]
        
    videolist = glob.glob(path[:path.find(videoName)] + "/*")
    print(fileName)

    print('Process all the files? (yes, no) :', end=" ")
    flag = input() 

    print('select windowsize:')
    winsize = list(map(str, input().split()))

    if flag == 'yes':
        for i in videolist:
            videoName = i[i.rfind('/')+1:]
            if not os.path.isdir(i + '/video'):
                make_dirs.makeDir(i)
            for ws in winsize:
                savepath = i + '/video'
                path = i + '/winsize/winsize_' + ws + '/pointData_' + videoName + '.txt'

                dataName = path[path.rfind('/')+1:-4]
                videoPath = path[:path.find('/opticalflow')] + '/video/yohaku/' + dirName + '/' + videoName + '.avi'

                print(savepath, path, videoPath)
                
                if os.path.isdir(i + '/winsize'):
                    doGet(path, savepath, videoPath, ws)
    else:
        for ws in winsize:
            savepath = path[:path.rfind('/')]
            print(savepath, path)

            dataName = path[path.rfind('/')+1:-4]
            videoPath = 'D:/opticalflow/video' + path[path.find('point_data')+10:path.find(dataName)-1] + '.avi'
            videoName = videoPath[videoPath.rfind('/')+1:]

                
            doGet(path, savepath, videoPath, ws)