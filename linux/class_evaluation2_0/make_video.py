def doGet(path, savepath, dirName):
    import cv2
    import pickle
    import numpy as np

    # 動画読み込み
    cap = cv2.VideoCapture(videoPath)

    # 座標ファイル読み込み
    f = open(path, 'rb')
    zahyou = pickle.load(f)

    # 動画読み込み設定
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    print(savepath + '/' + videoName)
    writer = cv2.VideoWriter(savepath + '/' + videoName, fourcc, fps, (width, height))

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

    typ = [('','*')] 
    dir = 'C:\\pg'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
    videoDir = path[:path.rfind('/')]
        
    videolist = glob.glob(videoDir[:videoDir.rfind('/')] + "/*")

    print('Process all the files? (yes, no) :', end=" ")
    flag = input() 

    if flag == 'yes':
        for i in videolist:
            savepath = i
            dirName = videoDir[videoDir.rfind('/', 0, videoDir.rfind('/'))+1:]
            path = i + '/pointData_' + i[i.rfind('\\')+1:] + '.txt'

            dataName = path[path.rfind('/')+1:-4]
            videoPath = 'D:/opticalflow/video' + path[path.find('point_data')+10:path.find(dataName)-1] + '.avi'
            videoName = videoPath[videoPath.rfind('\\')+1:]

            print(savepath, path)
            
            doGet(path, savepath, dirName)
    else:
        savepath = path[:path.rfind('/')]
        dirName = videoDir[videoDir.rfind('/', 0, videoDir.rfind('/'))+1:]
        print(savepath, path)

        dataName = path[path.rfind('/')+1:-4]
        videoPath = 'D:/opticalflow/video' + path[path.find('point_data')+10:path.find(dataName)-1] + '.avi'
        videoName = videoPath[videoPath.rfind('/')+1:]

            
        doGet(path, savepath, dirName)