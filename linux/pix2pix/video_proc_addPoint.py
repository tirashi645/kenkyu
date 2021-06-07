import numpy as np
import cv2
import os
import glob
import pickle
import sys
from pythonFile import proc, removeNoise, labeling, get_keypoint, image_keypoint
from PIL import Image
from keras.preprocessing.image import img_to_array
sys.path.insert(1, '/media/koshiba/Data/simple-HRNet')
from SimpleHRNet import SimpleHRNet

def todo(path):

    padding = 10    # 特徴点検出領域の半径
    p = 10
    nchannels=48
    njoints=17
    c = [[255, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255]]    # 特徴点の色

    kernel = np.ones((3,3),np.uint8)

    # Creat the HRNet model
    model = SimpleHRNet(nchannels, njoints, "/media/koshiba/Data/simple-HRNet/weights/pose_hrnet_w48_384x288.pth")

    # 読み込む動画の設定
    videoName = path.split('/')[-1][:-4]
    savePath = '/media/koshiba/Data/pix2pix/output/proc_feature'
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
        maxCorners=511,            # 保持するコーナー数,int
        qualityLevel=0.2,          # 最良値(最大個数値の割合),double
        minDistance=1,             # この距離内のコーナーを棄却,double
        blockSize=3,               # 使用する近傍領域のサイズ,int
        useHarrisDetector=False,   # FalseならShi-Tomashi法
        # k=0.04,　　　　　　　　　# Harris法の測度に使用
    )

    # 最初のフレームを読み込む
    ret, first_frame = cap.read()
    if rot==1:
        first_frame = np.rot90(first_frame, -1)
    #HRNetで骨格点を推測
    pts = model.predict(first_frame)
    #グレースケール変換
    pil_img = Image.fromarray(first_frame)
    org_img = img_to_array(pil_img)
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # マスク画像の生成
    #mask_img = proc.video_proc(pil_img)
    gen_img = proc.video_proc_gray(pil_img)     # この中でグレースケール化してる
    # ノイズを除去してセグメントを膨張する
    mask_img, img_mask = removeNoise.todo(gen_img)
    mask_img = labeling.remove_noise(mask_img)              # ラベリング処理
    mask_img = cv2.erode(mask_img,kernel,iterations = 1)    # 縮小処理
    #pts, keypoint_img = image_keypoint.get_keypoint(first_frame, mask_img)
    #print(pts)

    # 読み込んだフレームの特徴点を探す
    prev_points = cv2.goodFeaturesToTrack(
        image=first_gray,      # 入力画像
        mask=None,            # mask=0のコーナーを無視
        **feature_params
    )
    flow_layer = np.zeros_like(first_frame)
    flow_layer2 = np.zeros_like(first_frame)
    skelton_frame = first_frame.copy()
    point_frame = first_frame.copy()
    # 一度すべての点をノイズとする
    noise = [0 for i in range(len(prev_points))]

    feature_num = [6, 11, 12, 15, 16]
    feature_value = [100000, 100000, 100000, 100000, 100000, 0]
    feature_point = [-1,-1,-1,-1,-1,-1]     #右肩，左腰，右腰，左足，右足, 銃
    shooter_pt = []

    
    for pt in pts:
        cnt = 0
        for i,data in enumerate(pt):
            h = int(data[0])
            w = int(data[1])
            if mask_img[h][w] == 255:
                cnt+=1
            if cnt==13:
                skelton_frame = draw(skelton_frame, pt)
                shooter_pt = pt
                break
    print(shooter_pt)
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
            for p, num1 in enumerate(feature_num):
                a = np.array([shooter_pt[num1][1], shooter_pt[num1][0]])
                b = np.array([i[0][0], i[0][1]])
                u = np.linalg.norm(b - a)
                if u<feature_value[p]:
                    feature_point[p] = num
                    feature_value[p] = u
            if feature_value[-1]<i[0][0]:
                feature_point[-1] = num
                feature_value[-1] = i[0][0]

        else:
            flow_layer = cv2.circle(
                                            flow_layer,     # 描く画像
                                            (x, y),         # 線を引く始点
                                            2,              # 線を引く終点
                                            color = c[0],   # 描く色 青
                                            thickness=3     # 線の太さ
                                        )

    frame = cv2.add(skelton_frame, flow_layer)
    point_frame = cv2.add(point_frame, flow_layer)
    for num, i in enumerate(feature_point):
        x = int(prev_points[i][0][0])
        y = int(prev_points[i][0][1])
        #if max(gen_img[y][x]) > 255/2:
        if mask_img[y][x] == 255:
            flow_layer2 = cv2.circle(
                                            flow_layer2,     # 描く画像
                                            (x, y),         # 線を引く始点
                                            2,              # 線を引く終点
                                            color = c[1],   # 描く色 赤
                                            thickness=5     # 線の太さ
                                        )
        else:
            flow_layer2 = cv2.circle(
                                            flow_layer2,     # 描く画像
                                            (x, y),         # 線を引く始点
                                            2,              # 線を引く終点
                                            color = c[0],   # 描く色 青
                                            thickness=5     # 線の太さ
                                        )

    frame2 = cv2.add(first_frame, flow_layer2)
    print(feature_point, feature_value)

    if not os.path.exists(savePath + '/' + videoName):
        os.makedirs(savePath + '/' + videoName)
    cv2.imwrite(savePath + '/' + videoName + '/gen_' + videoName + '.jpg', gen_img)
    cv2.imwrite(savePath + '/' + videoName + '/mask_' + videoName + '.jpg', mask_img)
    cv2.imwrite(savePath + '/' + videoName + '/filter_' + videoName + '.jpg', img_mask)
    cv2.imwrite(savePath + '/' + videoName + '/point_' + videoName + '.jpg', point_frame)
    cv2.imwrite(savePath + '/' + videoName + '/main_' + videoName + '.jpg', frame)
    cv2.imwrite(savePath + '/' + videoName + '/track_' + videoName + '.jpg', frame2)
    cv2.imwrite(savePath + '/' + videoName + '/org_' + videoName + '.jpg', org_img)
    cv2.imwrite(savePath + '/' + videoName + '/gray_' + videoName + '.jpg', first_gray)
    #cv2.imwrite(savePath + '/' + videoName + '/keypoint_' + videoName + '.jpg', keypoint_img)
    '''
    with open(savePath + '/' + videoName + '/data_' + videoName + '.pickle', 'wb') as f:
        pickle.dump(evalute_list, f)
    '''
    #return evalute_list

def draw(image, pt):
    # Model parameters
    nchannels=48
    njoints=17
    line = [[1,2],[1,3],[1,6],[1,7],[6,7],[3,5],[2,4],[6,8],[6,12],[7,9],[7,13],[8,10],[9,11],[12,13],[12,14],[13,15],[14,16],[15,17]]
    cp = ['p' if i[2]>0.5 else 'f' for i in pt]
    #person_ids = np.arange(len(pts), dtype=np.int32)
    
    #frame = draw_points_and_skeleton(image, pt, joints_dict()['coco']['skeleton'], person_index=0, points_color_palette='gist_rainbow', skeleton_color_palette='jet',points_palette_samples=10)
    for i,data in enumerate(pt):
        h = int(data[0])
        w = int(data[1])
        #print(w, h)
        if cp[i]=='p':
            cv2.circle(image, (w, h), 5, (255, 255, 255), thickness=-1)
            cv2.putText(image, str(i+1), (w, h), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
    for point in line:
        s = point[0]-1
        t = point[1]-1
        if cp[s]=='p' and cp[t]=='p':
            cv2.line(image, (int(pt[s][1]), int(pt[s][0])), (int(pt[t][1]), int(pt[t][0])), (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    return image

if __name__=='__main__':

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
        todo(path)
        '''
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
    '''