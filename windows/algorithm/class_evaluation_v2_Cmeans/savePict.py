def todo(path, classlist, original, m, u):
    import numpy as np
    import cv2
    from pythonFile import click_pct
    import os

    padding = 10    # 特徴点検出領域の半径

    # 読み込む動画の設定
    videoName = path[path.rfind('/')+1:]
    cap = cv2.VideoCapture(path)
    print(videoName)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    rot = 0
    if width>height:
        rot = 1
        tmp = width
        width = height
        height = tmp

    # Shi-Tomashiのコーナー検出パラメータ
    feature_params = dict(
        maxCorners=255,            # 保持するコーナー数,int
        qualityLevel=0.2,          # 最良値(最大個数値の割合),double
        minDistance=7,             # この距離内のコーナーを棄却,double
        blockSize=7,               # 使用する近傍領域のサイズ,int
        useHarrisDetector=False,   # FalseならShi-Tomashi法
        # k=0.04,　　　　　　　　　# Harris法の測度に使用
    )

    # Lucas-Kanada法のパラメータ
    lk_params = dict(
        winSize=(15,15),
        maxLevel=2,

        #検索を終了する条件
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03
        ),

        # 測定値や固有値の使用
        flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
    )

    # 最初のフレームを読み込む
    ret, first_frame = cap.read()
    if rot==1:
        first_frame = np.rot90(first_frame, -1)

    #グレースケール変換
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # 読み込んだフレームの特徴点を探す
    prev_points = cv2.goodFeaturesToTrack(
        image=first_gray,      # 入力画像
        mask=None,            # mask=0のコーナーを無視
        **feature_params
    )

    flow_layer = np.zeros_like(first_frame)
    flow_layer0 = np.zeros_like(first_frame)
    flow_layer1 = np.zeros_like(first_frame)
    flow_layer2 = np.zeros_like(first_frame)
    classNum = 1

    print(type(u))
    u = (np.array(u) - 0.33) / 0.67 * 255

    for num, classData in enumerate(classlist):
        for index, prev in enumerate(prev_points):
            #if original[index]!=classData[index]:
            if classData[index]==2:
                flow_layer = cv2.circle(
                                            flow_layer,                               # 描く画像
                                            (int(prev[0][0]), int(prev[0][1])),         # 線を引く始点
                                            5,         # 線を引く終点
                                            color = (u[num][index][2], u[num][index][1], u[num][index][0]),    # 描く色
                                            thickness=3   # 線の太さ
                                        )
                flow_layer0 = cv2.circle(
                                            flow_layer0,                               # 描く画像
                                            (int(prev[0][0]), int(prev[0][1])),         # 線を引く始点
                                            5,         # 線を引く終点
                                            color = (u[num][index][2], u[num][index][1], u[num][index][0]),    # 描く色
                                            thickness=3   # 線の太さ
                                        )
            elif classData[index]==1:
                flow_layer = cv2.circle(
                                            flow_layer,                               # 描く画像
                                            (int(prev[0][0]), int(prev[0][1])),         # 線を引く始点
                                            5,         # 線を引く終点
                                            color = (u[num][index][2], u[num][index][1], u[num][index][0]),    # 描く色
                                            thickness=3   # 線の太さ
                                        )
                flow_layer1 = cv2.circle(
                                            flow_layer1,                               # 描く画像
                                            (int(prev[0][0]), int(prev[0][1])),         # 線を引く始点
                                            5,         # 線を引く終点
                                            color = (u[num][index][2], u[num][index][1], u[num][index][0]),    # 描く色
                                            thickness=3   # 線の太さ
                                        )
            elif classData[index]==0:
                flow_layer = cv2.circle(
                                            flow_layer,                               # 描く画像
                                            (int(prev[0][0]), int(prev[0][1])),         # 線を引く始点
                                            5,         # 線を引く終点
                                            color = (u[num][index][2], u[num][index][1], u[num][index][0]),    # 描く色
                                            thickness=3   # 線の太さ
                                        )
                flow_layer2 = cv2.circle(
                                            flow_layer2,                               # 描く画像
                                            (int(prev[0][0]), int(prev[0][1])),         # 線を引く始点
                                            5,         # 線を引く終点
                                            color = (u[num][index][2], u[num][index][1], u[num][index][0]),    # 描く色
                                            thickness=3   # 線の太さ
                                        )
            elif classData[index]==-1:
                flow_layer = cv2.circle(
                                            flow_layer,                               # 描く画像
                                            (int(prev[0][0]), int(prev[0][1])),         # 線を引く始点
                                            5,         # 線を引く終点
                                            color = (0, 0, 0),    # 描く色
                                            thickness=3   # 線の太さ
                                        )
        frame2 = cv2.add(first_frame, flow_layer)
        frame_class0 = cv2.add(first_frame, flow_layer0)
        frame_class1 = cv2.add(first_frame, flow_layer1)
        frame_class2 = cv2.add(first_frame, flow_layer2)

        # 結果画像の表示
        #cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        #cv2.imshow("frame", frame2)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite('D:/opticalflow/cmeans/result/class' + str(classNum) + '/' + str(videoName[:-4]) + '_' + str(m) + '_original.jpg', frame2)
        cv2.imwrite('D:/opticalflow/cmeans/result/class' + str(classNum) + '/' + str(videoName[:-4]) + '_' + str(m) + '_c1.jpg', frame_class0)
        cv2.imwrite('D:/opticalflow/cmeans/result/class' + str(classNum) + '/' + str(videoName[:-4]) + '_' + str(m) + '_c2.jpg', frame_class1)
        cv2.imwrite('D:/opticalflow/cmeans/result/class' + str(classNum) + '/' + str(videoName[:-4]) + '_' + str(m) + '_c3.jpg', frame_class2)
        classNum += 1
        print('finish')


if __name__=='__main__':
    from tkinter import filedialog
    from pythonFile import click_pct, timestump, k_means
    import glob
    import os
    # ファイルダイアログからファイル選択
    typ = [('','*')] 
    dir = 'C:\\pg'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 
    noizu = todo(path, [[0],[0],[0]])