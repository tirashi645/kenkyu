def todo(path):
    import numpy as np
    import cv2
    from pythonFile import click_pct
    import pickle
    import os

    padding = 5    # 特徴点検出領域の半径
    save_path = 'E:/data/click_point/nda/output'

    # 読み込む動画の設定
    path = path.replace(os.sep, '/')
    videoName = path.split('/')[-1][:path.split('/')[-1].rfind('.')]
    videoDir = path.split('/')[-2]
    cap = cv2.VideoCapture(path)
    print(videoName)

    # 動画の設定
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    rot = 0
    if width>height:
        rot = 1
        tmp = width
        width = height
        height = tmp

    print(videoName[:-4])

    # Shi-Tomashiのコーナー検出パラメータ
    feature_params = dict(
        maxCorners=255,            # 保持するコーナー数,int
        qualityLevel=0.2,          # 最良値(最大個数値の割合),double
        minDistance=7,             # この距離内のコーナーを棄却,double
        blockSize=7,               # 使用する近傍領域のサイズ,int
        useHarrisDetector=False,   # FalseならShi-Tomashi法
        # k=0.04,　　　　　　　　　# Harris法の測度に使用
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
    # 一度すべての点をノイズとする
    noise = [0] * len(prev_points)
    hozon = noise

    for i in prev_points:
        flow_layer = cv2.circle(
                                        flow_layer,                           # 描く画像
                                        (int(i[0][0]), int(i[0][1])),         # 線を引く始点
                                        2,         # 線を引く終点
                                        color = (0, 0, 255),    # 描く色
                                        thickness=3   # 線の太さ
                                    )
    frame = cv2.add(first_frame, flow_layer)
    frame = cv2.resize(frame, (720, 1280))

    #######################################
    # クリックした特徴点を正常な特徴点とする      
    #######################################

    while True:
        # クリックした座標を保存
        ret, points = click_pct.give_coorList(frame)
        points = np.array(points, dtype='int')
        # クリックした座標の周囲の点を正常な特徴点とする
        for p in points:
            area = [p[0]-padding, p[0]+padding, p[1]-padding, p[1]+padding]
            for index, prev in enumerate(prev_points):
                if (area[0]<=int(prev[0][0]))and(area[1]>=int(prev[0][0]))and(area[2]<=int(prev[0][1]))and(area[3]>=int(prev[0][1])):
                    if noise[index]==0:
                        noise[index] = 1
                    elif noise[index]==1:
                        noise[index] = 0

        flow_layer2 = np.zeros_like(first_frame)
        for index, prev in enumerate(prev_points):
            if noise[index]==0:
                flow_layer2 = cv2.circle(
                                                flow_layer2,                               # 描く画像
                                                (int(prev[0][0]), int(prev[0][1])),         # 線を引く始点
                                                5,         # 線を引く終点
                                                color = (0, 0, 255),    # 描く色
                                                thickness=3   # 線の太さ
                                            )
            elif noise[index]==1:
                flow_layer2 = cv2.circle(
                                                flow_layer2,                               # 描く画像
                                                (int(prev[0][0]), int(prev[0][1])),         # 線を引く始点
                                                5,         # 線を引く終点
                                                color = (255, 0, 0),    # 描く色
                                                thickness=3   # 線の太さ
                                            )
        frame = cv2.add(first_frame, flow_layer2)
        if ret==1:
            break

    # 結果画像の表示
    #cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    #cv2.imshow("frame", frame)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    category = np.array(noise)
    if not os.path.exists(save_path + '/' + videoDir):
        os.makedirs(save_path + '/' + videoDir)
    cv2.imwrite(save_path + '/' + videoDir + '/' + str(videoName) + '_Original.jpg', frame)
    #with open(save_path + '/' + videoDir + '_' + str(videoName) + '.pickle', 'wb') as f:
    #    pickle.dump(category, f)
    with open(save_path + '/' + str(videoName) + '.pickle', 'wb') as f:
        pickle.dump(category, f)

    return noise

if __name__=='__main__':
    from tkinter import filedialog
    from pythonFile import click_pct, timestump, k_means
    import glob
    import os
    # ファイルダイアログからファイル選択
    video_file = glob.glob('E:/data/click_point/nda/input/*')

    for path in video_file:
        noizu = todo(path)