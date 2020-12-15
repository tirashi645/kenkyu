def todo(path, zahyou):
    import numpy as np
    import cv2
    from pythonFile import click_pct
    import os

    padding = 10    # 特徴点検出領域の半径
    c = [[255, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255]]    # 特徴点の色
    selectDir = ['cat1', 'cat2', 'cat3', 'cat4']

    # 読み込む動画の設定
    videoDir = path[:path.rfind('/')]
    dirName = videoDir[videoDir.rfind('/')+1:]
    videoName = path[path.rfind('/')+1:-4]
    savePath = 'D:/opticalflow/point_data/' + dirName + '/' + videoName
    cap = cv2.VideoCapture(path)
    print(path[path.rfind('/')+1:])

    # 動画の設定
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

    print(videoName)

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

    for i in prev_points:
        flow_layer = cv2.circle(
                                        flow_layer,                           # 描く画像
                                        (int(i[0][0]), int(i[0][1])),         # 線を引く始点
                                        2,         # 線を引く終点
                                        color = c[0],    # 描く色
                                        thickness=3   # 線の太さ
                                    )
    frame = cv2.add(first_frame, flow_layer)
    save_layer_list = [np.zeros_like(first_frame)] * len(zahyou)
    selectDirList = ['/cat1/'] * len(zahyou)

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
                    if (noise[index]==0):
                        noise[index] = 1
                        zahyou[index][0][2] = 1
                        #break
                    elif (noise[index]==1):
                        noise[index] = 2
                        zahyou[index][0][2] = 2
                        #break
                    elif (noise[index]==2):
                        noise[index] = 3
                        zahyou[index][0][2] = 3
                        #break
                    elif (noise[index]==3):
                        noise[index] = 0
                        zahyou[index][0][2] = 0
                        #break

        # 特徴点を描画するlayer
        cat1_layer = np.zeros_like(first_frame)
        cat2_layer = np.zeros_like(first_frame)
        cat3_layer = np.zeros_like(first_frame)
        cat4_layer = np.zeros_like(first_frame)

        for index, prev in enumerate(prev_points):
            save_layer = np.zeros_like(first_frame)
            save_layer_list[index] = cv2.circle(
                                            save_layer,                               # 描く画像
                                            (int(prev[0][0]), int(prev[0][1])),         # 線を引く始点
                                            5,         # 線を引く終点
                                            color = c[noise[index]],    # 描く色
                                            thickness=3   # 線の太さ
                                        )
            cat1_layer = cv2.circle(
                                            cat1_layer,                               # 描く画像
                                            (int(prev[0][0]), int(prev[0][1])),         # 線を引く始点
                                            5,         # 線を引く終点
                                            color = c[noise[index]],    # 描く色
                                            thickness=3   # 線の太さ
                                        )
            selectDirList[index] = '/cat' + str(noise[index]+1) + '/'
        # フレームに特徴点layerを追加する
        frame = cv2.add(first_frame, cat1_layer)
        frame = cv2.add(frame, cat2_layer)
        frame = cv2.add(frame, cat3_layer)
        frame = cv2.add(frame, cat4_layer)
        layer_list = [cat1_layer, cat2_layer, cat3_layer, cat4_layer]
        if ret==1:
            break

    # 結果画像の表示
    #cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(savePath + '/' + videoName + '.jpg', frame)

    # 分類結果の画像を保存する
    for n,layer in enumerate(layer_list):
        frame = cv2.add(first_frame, layer)
        cv2.imwrite(savePath + '/' + selectDir[n] + '/' + videoName + '_' + selectDir[n] + '.jpg', frame)

    for index in range(len(zahyou)):
        save_frame = cv2.add(first_frame, save_layer_list[index])
        cv2.imwrite(savePath + '/' + selectDirList[index] + '/pict/' + videoName + '_' + str(index+1) + '.jpg', save_frame)

    return zahyou

if __name__=='__main__':
    pass
    '''
    from tkinter import filedialog
    from pythonFile import click_pct, timestump, k_means
    import glob
    import os
    # ファイルダイアログからファイル選択
    typ = [('','*')] 
    dir = 'C:\\pg'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 
    noizu = todo(path, )
    '''