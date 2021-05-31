def todo(path):
    import numpy as np
    import cv2
    import os
    import pickle
    from pythonFile import proc, removeNoise, labeling
    from PIL import Image
    from keras.preprocessing.image import img_to_array

    padding = 10    # 特徴点検出領域の半径
    p = 10
    c = [[255, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255]]    # 特徴点の色
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    kernel = np.ones((5,5),np.uint8)

    # 読み込む動画の設定
    videoName = path.split('/')[-1][:-4]
    savePath = '/media/koshiba/Data/pix2pix/output/proc_point2'
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

    # マスク画像の生成
    #mask_img = proc.video_proc(pil_img)
    gen_img = proc.video_proc_gray(pil_img)     # この中でグレースケール化してる
    # ノイズを除去してセグメントを膨張する
    mask_img, img_mask = removeNoise.todo(gen_img)
    mask_img = labeling.remove_noise(mask_img)
    mask_img = cv2.erode(mask_img,kernel,iterations = 1)

    # 読み込んだフレームの特徴点を探す
    prev_points = cv2.goodFeaturesToTrack(
        image=first_gray,      # 入力画像
        mask=None,            # mask=0のコーナーを無視
        **feature_params
    )
    flow_layer = np.zeros_like(first_frame)
    flow_layer2 = np.zeros_like(first_frame)
    # 一度すべての点をノイズとする
    noise = [0 for i in range(len(prev_points))]

    # マスク画像内の特徴点を探す
    for num, i in enumerate(prev_points):
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
            noise[num] = 1
        else:
            flow_layer = cv2.circle(
                                            flow_layer,     # 描く画像
                                            (x, y),         # 線を引く始点
                                            2,              # 線を引く終点
                                            color = c[0],   # 描く色 青
                                            thickness=3     # 線の太さ
                                        )
    frame = cv2.add(first_frame, flow_layer)



    # 評価計算
    point_evalute = []
    if os.path.exists('/media/koshiba/Data/pix2pix/input/point/' + videoName + '.pickle'):
        with open('/media/koshiba/Data/pix2pix/input/point/' + videoName + '.pickle', 'rb') as f:
            point_data = pickle.load(f)
            for i in range(len(point_data)):
                if point_data[i]==1 and noise[i]==1:
                    tp+=1
                    point_evalute.append('tp')
                elif point_data[i]==1 and noise[i]==0:
                    fn+=1
                    point_evalute.append('fn')
                elif point_data[i]==0 and noise[i]==0:
                    tn+=1
                    point_evalute.append('tn')
                elif point_data[i]==0 and noise[i]==1:
                    fp+=1
                    point_evalute.append('fp')

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        specificity = tn/(fp+tn)
        f_value = (2 * recall * precision) / (recall + precision)

        print('acc:{:.3f}, pre:{:.3f}, rec:{:.3f}, spe:{:.3f}, f_value:{:.3f}'.format(accuracy, precision, recall, specificity, f_value))
    value_list = [accuracy, precision, recall, specificity, f_value]
    
    evalute_list = [value_list, point_data]

    for num, i in enumerate(prev_points):
        x = int(i[0][0])
        y = int(i[0][1])
        #if max(gen_img[y][x]) > 255/2:
        if point_evalute[num]=='tp':
            flow_layer2 = cv2.circle(
                                            flow_layer,     # 描く画像
                                            (x, y),         # 線を引く始点
                                            2,              # 線を引く終点
                                            color = c[1],   # 描く色 赤
                                            thickness=3     # 線の太さ
                                        )
            noise[num] = 1
        elif point_evalute[num]=='tn':
            flow_layer2 = cv2.circle(
                                            flow_layer,     # 描く画像
                                            (x, y),         # 線を引く始点
                                            2,              # 線を引く終点
                                            color = c[0],   # 描く色 青
                                            thickness=3     # 線の太さ
                                        )
        elif point_evalute[num]=='fp':
            flow_layer2 = cv2.circle(
                                            flow_layer,     # 描く画像
                                            (x, y),         # 線を引く始点
                                            2,              # 線を引く終点
                                            color = c[2],   # 描く色 青
                                            thickness=3     # 線の太さ
                                        )
        elif point_evalute[num]=='fn':
            flow_layer2 = cv2.circle(
                                            flow_layer,     # 描く画像
                                            (x, y),         # 線を引く始点
                                            2,              # 線を引く終点
                                            color = c[3],   # 描く色 青
                                            thickness=3     # 線の太さ
                                        )
    frame2 = cv2.add(first_frame, flow_layer2)
    frame3 = frame2
    txt = 'acc:{:.3f}, spe:{:.3f}, f_value:{:.3f}'.format(accuracy, specificity, f_value)
    cv2.putText(frame3, txt, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 1, cv2.LINE_AA)

    if not os.path.exists(savePath + '/' + videoName):
        os.makedirs(savePath + '/' + videoName)
    cv2.imwrite(savePath + '/' + videoName + '/gen_' + videoName + '.jpg', gen_img)
    cv2.imwrite(savePath + '/' + videoName + '/mask_' + videoName + '.jpg', mask_img)
    cv2.imwrite(savePath + '/' + videoName + '/filter_' + videoName + '.jpg', img_mask)
    cv2.imwrite(savePath + '/' + videoName + '/' + videoName + '.jpg', frame)
    cv2.imwrite(savePath + '/' + videoName + '/' + videoName + '_evalute.jpg', frame2)
    cv2.imwrite(savePath + '/' + videoName + '/org_' + videoName + '.jpg', org_img)
    cv2.imwrite(savePath + '/' + videoName + '/gray_' + videoName + '.jpg', first_gray)
    cv2.imwrite('/media/koshiba/Data/pix2pix/output/proc_pict/' + videoName + '.jpg', frame3)
    with open(savePath + '/' + videoName + '/data_' + videoName + '.pickle', 'wb') as f:
        pickle.dump(evalute_list, f)
    return evalute_list


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

    accuracy = 0
    precision = 0
    recall = 0
    specificity = 0
    f_value = 0

    acc_list = [-1.0, 10000.0]
    f_list = [-1.0, 10000.0]

    for path in video_file:

        evalute_list = todo(path)
        value_list = evalute_list[0]
        videoName = path.split('/')[-1][:-4]

        accuracy += value_list[0]
        precision += value_list[1]
        recall += value_list[2]
        specificity += value_list[3]
        f_value += value_list[4]
        if acc_list[0] < value_list[0]:
            max_acc = [videoName, value_list]
            acc_list[0] = value_list[0]
        if acc_list[1] > value_list[0]:
            min_acc = [videoName, value_list]
            acc_list[1] = value_list[0]

        if f_list[0] < value_list[4]:
            max_fValue = [videoName, value_list]
            f_list[0] = value_list[4]
        if f_list[1] > value_list[4]:
            min_fValue = [videoName, value_list]
            f_list[1] = value_list[4]

        

    accuracy /= len(video_file)
    precision /= len(video_file)
    recall /= len(video_file)
    specificity /= len(video_file)
    f_value /= len(video_file)

    value_list = [accuracy, precision, recall, specificity, f_value]
    minmax = [max_acc, min_acc, max_fValue, min_fValue]

    with open('/media/koshiba/Data/pix2pix/output/proc_point/evalute.pickle', 'wb') as f:
        pickle.dump(video_file, f)
    with open('/media/koshiba/Data/pix2pix/output/proc_point/minmax.pickle', 'wb') as f:
        pickle.dump(minmax, f)

    print('------------------------------')
    print('acc:{:.3f}, pre:{:.3f}, rec:{:.3f}, spe:{:.3f}, f_value:{:.3f}'.format(accuracy, precision, recall, specificity, f_value))
    print(acc_list,f_list)
    print('max_acc-----------------------')
    print(max_acc)
    print('min_acc-----------------------')
    print(min_acc)
    print('max_fValue--------------------')
    print(max_fValue)
    print('min_fValue--------------------')
    print(min_fValue)