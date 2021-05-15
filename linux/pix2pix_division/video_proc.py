def todo(path):
    import numpy as np
    import cv2
    import os
    import pickle
    import proc, removeNoise, labeling
    from PIL import Image
    from keras.preprocessing.image import img_to_array

    padding = 10    # 特徴点検出領域の半径
    p = 10
    c = [[255, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255]]    # 特徴点の色

    # 読み込む動画の設定
    videoName = path.split('/')[-1][:-4]
    savePath = '/media/koshiba/Data/pix2pix/output/proc_point'
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
    pil_img = Image.fromarray(first_frame)
    org_img = img_to_array(pil_img)
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    upper_img = pil.crop((0, 0, width, height/2))
    lower_img = pil.crop((0, height/2, width, height))

    # マスク画像の生成
    #mask_img = proc.video_proc(pil_img)
    gen_upper = proc.video_proc_gray(upper_img)     # この中でグレースケール化してる
    gen_lower = proc.video_proc_gray(lower_img)

    gen_image = cv2.vconcat([gen_upper, lower_img])

    # ノイズを除去してセグメントを膨張する
    mask_img, img_mask = removeNoise.todo(gen_image)
    mask_img = labeling.remove_noise(mask_img)

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
        x = int(i[0][0])
        y = int(i[0][1])
        #if max(gen_img[y][x]) > 255/2:
        if mask_img[y][x] == 255:
            flow_layer = cv2.circle(
                                            flow_layer,     # 描く画像
                                            (x, y),         # 線を引く始点
                                            2,              # 線を引く終点
                                            color = c[1],   # 描く色 赤
                                            thickness=3     # 線の太さ
                                        )
        else:
            flow_layer = cv2.circle(
                                            flow_layer,     # 描く画像
                                            (x, y),         # 線を引く始点
                                            2,              # 線を引く終点
                                            color = c[0],   # 描く色 青
                                            thickness=3     # 線の太さ
                                        )
    frame = cv2.add(first_frame, flow_layer)
    if not os.path.exists(savePath + '/' + videoName):
        os.makedirs(savePath + '/' + videoName + '/division')
    cv2.imwrite(savePath + '/' + videoName + '/division' + '/gen_' + videoName + '.jpg', gen_image)
    cv2.imwrite(savePath + '/' + videoName + '/division' + '/mask_' + videoName + '.jpg', mask_img)
    cv2.imwrite(savePath + '/' + videoName + '/division' + '/filter_' + videoName + '.jpg', img_mask)
    cv2.imwrite(savePath + '/' + videoName + '/division' + '/' + videoName + '.jpg', frame)
    cv2.imwrite(savePath + '/' + videoName + '/division' + '/org_' + videoName + '.jpg', org_img)
    cv2.imwrite(savePath + '/' + videoName + '/division' + '/gray_' + videoName + '.jpg', first_gray)


if __name__=='__main__':
    import glob
    import os
    import pickle

    # ファイルダイアログからファイル選択
    '''
    typ = [('','*')] 
    dir = 'C:\\pg'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 
    '''
    video_path = '/media/koshiba/Data/pix2pix/input/video'
    video_file = glob.glob(video_path + '/*')

    for path in video_file:

        todo(path)