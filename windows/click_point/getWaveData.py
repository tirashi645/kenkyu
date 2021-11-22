import cv2
import sys
import numpy as np
from tkinter import filedialog
import copy
import matplotlib.pyplot as plt

clickPoint = [0, 0]
flag = 0

class mouseParam:
    def __init__(self, input_img_name):
        #マウス入力用のパラメータ
        self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
        #マウス入力の設定
        cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)
    
    #コールバック関数
    def __CallBackFunc(self, eventType, x, y, flags, userdata):
        
        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType    
        self.mouseEvent["flags"] = flags    

    #マウス入力用のパラメータを返すための関数
    def getData(self):
        return self.mouseEvent
    
    #マウスイベントを返す関数
    def getEvent(self):
        return self.mouseEvent["event"]                

    #マウスフラグを返す関数
    def getFlags(self):
        return self.mouseEvent["flags"]                

    #xの座標を返す関数
    def getX(self):
        return self.mouseEvent["x"]  

    #yの座標を返す関数
    def getY(self):
        return self.mouseEvent["y"]  

    #xとyの座標を返す関数
    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])
    
def give_coorList(f):

    clickList = []

    #表示するWindow名
    window_name = "frame"
    
    #画像の表示
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, f)
    
    #コールバックの設定
    mouseData = mouseParam(window_name)
    
    while 1:
        cv2.waitKey(20)
        # 左クリックがあったら座標をリストに入れる
        if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
            m_point = mouseData.getPos()
            # 座標がリストになければ追加する
            if clickList.count(m_point)==0:
                clickList.append(m_point)
            print(mouseData.getPos())
            flag = 1
            break
        #右クリックがあったら終了
        elif mouseData.getEvent() == cv2.EVENT_RBUTTONDOWN:
            flag = 2
            break
    cv2.destroyAllWindows()
    # 座標のリストを返す        
    return flag, clickList

def nothing(x):
    pass
    
if __name__ == '__main__':
    # ファイルダイアログからファイル選択
    typ = [('','*')] 
    dir = 'C:\\pg'
    image_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
    file_path = image_path 
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        sys.exit()
        
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
    # whileリープで読み込むための準備
    old_frame = first_frame
    old_gray = first_gray
    zahyou = prev_points.tolist() # 各フレームの特徴点の座標を入れておく
    FirstZahyou = copy.copy(zahyou)
    framelist = [1] # 各リストとフレーム数を同期させる
    # 2フレーム目以降でオプティカルフローを実行
    # zahyou: [x座標, y座標]
    while True:
        # 2枚目以降のフレームの読み込み
        ret, frame = cap.read()
        if ret==False:
            break
        if rot==1:
            frame = np.rot90(frame, -1)
        
        # グレースケール変換
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #オプティカルフロー（正確には対応点）の検出
        # next_points: 検出した対応点, numpy.ndarray
        # status: 各店において,見つかれば1(True),見つからなければ0(False), numpy.ndarray
        # err: 検出した点の誤差, numpy.ndarray
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            prevImg=old_gray,        # 前の画像(t-1)
            nextImg=frame_gray,      # 次の画像(t)
            prevPts=prev_points,     # 始点2次元ベクトル,特徴点やそれに準ずる店
            nextPts=None,            # 結果の2次元ベクトル
            **lk_params
        )

        # 正しく特徴点と対応点が検出できた点のみに絞る
        # todo: 数フレームおきに特徴点を検出しなおさないと，対応点がなくなるのでエラーになります
        good_new = next_points
        good_old = prev_points
        
        # 対応点の是票を保存
        for rank, (prev_p, next_p) in enumerate(zip(good_old, good_new)):
            
            # x,y座標の取り出し
            # prev_x, prev_y: numpy.float32
            # next_x, next_y: numpy.float32
            prev_x, prev_y = prev_p.flatten()
            next_x, next_y = next_p.flatten()
            
            # 座標保存
            zahyou[rank].append([next_x, next_y])
            
        
        # 次のフレームを読み込む準備
        old_gray = frame_gray.copy()
        prev_points = good_new.reshape(-1, 1, 2)
        result_img = frame
        framelist.append(1)

    cap.release()
    print(np.array(zahyou).shape)
    
    for i,point_data in enumerate(zahyou):
        vector1 = []    # 初期フレーム
        vector2 = []    # フレーム間
        vector_num = []
        gosa = []
        # a : 1フレーム前
        # a2 : 最初のフレーム
        # b : 現在のフレーム
        for index, point in enumerate(point_data):
            # 最初だけ読み込み
            if index == 0:
                a = np.array([point[0], point[1]])
                a2 = np.array([point[0], point[1]])
            # ２フレーム目以降読み込み
            elif index != 0:
                b = np.array([point[0], point[1]])
                # 最初のフレームと現在のフレームの誤差をリストに入れる
                gosa.append(abs(a2-b))
                # 初期フレームと現在のフレームの距離ベクトルをリストに入れる
                vector1.append(np.linalg.norm(b - a2))
                # 一つ前のフレームと現在のフレームの距離ベクトルをリストに入れる
                vector2.append(np.linalg.norm(b - a))
                vector_num.append(index)
                # 次のフレームの準備
                a = b
        
        bairitu = 10
        spines = 3*bairitu
        fig = plt.figure(figsize=(14*bairitu,10*bairitu))
        ax = fig.add_subplot(1, 1, 1)
        y = [f+1 for f in range(len(vector1))]

        # plot
        ax.plot(y, vector1, color='b', linewidth=(4*bairitu))
        
        #軸の太さの調整。方向を辞書のキーとして渡し、set_linewidthで大きさを微調整できる
        ax.spines["top"].set_linewidth(spines)
        ax.spines["left"].set_linewidth(spines)
        ax.spines["bottom"].set_linewidth(spines)
        ax.spines["right"].set_linewidth(spines)

        #plt.title('Method-2', fontsize=36)
        plt.xlabel('フレーム数', fontsize=36*bairitu, fontname ='MS Gothic')
        plt.ylabel('距離ベクトル', fontsize=36*bairitu, fontname ='MS Gothic')
        plt.tick_params(labelsize=36*bairitu)
        plt.savefig('D:/opticalflow/evaluation/vector/' + str(i) + '_wave.jpg')
        plt.close()
        '''
        px = int(point_data[0][0])
        py = int(point_data[0][1])
        p1 = max(0, py-20)
        p2 = min(height, py+20)
        p3 = max(0, px-20)
        p4 = min(width, px+20)
        cv2.imwrite('D:/opticalflow/evaluation/vector/' + str(i) + '_pict.jpg', first_frame[p1:p2][p3:p4])
        '''
        
                
    for i in FirstZahyou:
        flow_layer = cv2.circle(
                                        flow_layer,                           # 描く画像
                                        (int(i[0][0]), int(i[0][1])),         # 線を引く始点
                                        2,         # 線を引く終点
                                        color = (0, 0, 255),    # 描く色
                                        thickness=3   # 線の太さ
                                    )
    frame = cv2.add(first_frame, flow_layer)
    
    window_name = 'frame'
    window_wave = 'plot Wave'
    delay = 10
    cv2.namedWindow(window_name)
    cv2.namedWindow(window_wave)
    cv2.imshow(window_name, frame)
    while(True):
        flag = 0

        flag, points = give_coorList(frame)

        if flag:
            mindistance = 10000000
            pointNum = 0
            for num, p in enumerate(FirstZahyou):
                a = np.array(clickPoint)
                b = np.array(p[0])
                u = np.linalg.norm(b - a)
                #print(mindistance, num)
                if u<mindistance:
                    mindistance = u
                    pointNum = num
            print(pointNum)

            flow_layer2 = cv2.circle(
                                        flow_layer,                           # 描く画像
                                        (int(FirstZahyou[pointNum][0][0]), int(FirstZahyou[pointNum][0][1])),         # 線を引く始点
                                        2,         # 線を引く終点
                                        color = (0, 255, 0),    # 描く色
                                        thickness=3   # 線の太さ
                                    )
            frame = cv2.add(first_frame, flow_layer2)
            
        if flag==2:
            break