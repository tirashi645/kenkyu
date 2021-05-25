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

    # マスク画像の生成
    #mask_img = proc.video_proc(pil_img)
    gen_img = proc.video_proc_gray(pil_img)     # この中でグレースケール化してる
    # ノイズを除去してセグメントを膨張する
    mask_img, img_mask = removeNoise.todo(gen_img)
    mask_img = labeling.remove_noise(mask_img)

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

    tp = 0
    fp = 0
    tn = 0
    fn = 0
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

        print('acc:{:.3f}, pre:{:.3f}, rec:{:.3f}, spe:{:.3f}, f_value:{:.3f}'.format(accuracy, precision, recall, specificity, (2 * recall * precision) / (recall + precision)))
    value_list = [accuracy, precision, recall, specificity]
    
    evalute_list = [value_list, point_evalute]

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

    if not os.path.exists(savePath + '/' + videoName):
        os.makedirs(savePath + '/' + videoName)
    cv2.imwrite(savePath + '/' + videoName + '/gen_' + videoName + '.jpg', gen_img)
    cv2.imwrite(savePath + '/' + videoName + '/mask_' + videoName + '.jpg', mask_img)
    cv2.imwrite(savePath + '/' + videoName + '/filter_' + videoName + '.jpg', img_mask)
    cv2.imwrite(savePath + '/' + videoName + '/' + videoName + '.jpg', frame)
    cv2.imwrite(savePath + '/' + videoName + '/' + videoName + '_evalute.jpg', frame2)
    cv2.imwrite(savePath + '/' + videoName + '/org_' + videoName + '.jpg', org_img)
    cv2.imwrite(savePath + '/' + videoName + '/gray_' + videoName + '.jpg', first_gray)
    with open(savePath + '/' + videoName + '/data_' + videoName + '.pickle', 'wb') as f:
        pickle.dump(evalute_list, f)
    return value_list


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

    max_list = [0, 0]
    min_list = [100000, 100000]

    for path in video_file:

        value_list = todo(path)
        videoName = path.split('/')[-1][:-4]

        accuracy += value_list[0]
        precision += value_list[1]
        recall += value_list[2]
        specificity += value_list[3]
        f_tmp = (2 * recall * precision) / (recall + precision)

        if max_list[0] < value_list[0]:
            max_acc = [accuracy, precision, recall, specificity, videoName]
            max_list[0] = value_list[0]
        if min_list[0] > value_list[0]:
            min_acc = [accuracy, precision, recall, specificity, videoName]
            min_list[0] = value_list[0]

        if max_list[1] < f_tmp:
            max_fValue = [accuracy, precision, recall, specificity, videoName]
            max_list[1] = f_tmp
        if min_list[1] > f_tmp:
            min_fValue = [accuracy, precision, recall, specificity, videoName]
            min_list[1] = f_tmp

    accuracy /= len(video_file)
    precision /= len(video_file)
    recall /= len(video_file)
    specificity /= len(video_file)
    f_value = (2 * recall * precision) / (recall + precision)

    value_list = [accuracy, precision, recall, specificity, f_value]

    with open('/media/koshiba/Data/pix2pix/output/proc_point/evalute.pickle', 'wb') as f:
        pickle.dump(video_file, f)
    with open('/media/koshiba/Data/pix2pix/output/proc_point/max_acc.pickle', 'wb') as f:
        pickle.dump(max_acc, f)
    with open('/media/koshiba/Data/pix2pix/output/proc_point/mmin_acc.pickle', 'wb') as f:
        pickle.dump(min_acc, f)
    with open('/media/koshiba/Data/pix2pix/output/proc_point/max_fValue.pickle', 'wb') as f:
        pickle.dump(max_fValue, f)
    with open('/media/koshiba/Data/pix2pix/output/proc_point/min_fValue.pickle', 'wb') as f:
        pickle.dump(min_fValue, f)

    print('------------------------------')
    print('acc:{:.3f}, pre:{:.3f}, rec:{:.3f}, spe:{:.3f}, f_value:{:.3f}'.format(accuracy, precision, recall, specificity, f_value))
    print('max_acc-----------------------')
    print('acc:{:.3f}, spe:{:.3f}, f_value:{:.3f}'.format(map(str, max_acc[0], max_acc[3], max_acc[4])))
    print('min_acc-----------------------')
    print('acc:{:.3f}, spe:{:.3f}, f_value:{:.3f}'.format(map(str, min_acc[0], min_acc[3], min_acc[4])))
    print('max_fValue--------------------')
    print('acc:{:.3f}, spe:{:.3f}, f_value:{:.3f}'.format(map(str, max_fValue[0], max_fValue[3], max_fValue[4])))
    print('min_fValue--------------------')
    print('acc:{:.3f}, spe:{:.3f}, f_value:{:.3f}'.format(map(str, min_fValue[0], min_fValue[3], min_fValue[4])))