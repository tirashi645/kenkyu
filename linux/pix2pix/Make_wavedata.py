import numpy as np
import cv2
import pickle
import video_proc_addpoint

def todo(path):
    # 読み込む動画の設定
    videoDir = path[:path.rfind('/')]
    dirName = videoDir[videoDir.rfind('/')+1:]
    videoName = path[path.rfind('/')+1:-4]
    cap = cv2.VideoCapture(path)

    # 動画の設定を読み込み
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

    # Shi-Tomashiのコーナー検出パラメータ
    cnt, feature_point, feature_params = video_proc_addpoint.todo(path)
    '''
    feature_params = dict(
        maxCorners=255,            # 保持するコーナー数,int
        qualityLevel=0.2,          # 最良値(最大個数値の割合),double
        minDistance=7,             # この距離内のコーナーを棄却,double
        blockSize=7,               # 使用する近傍領域のサイズ,int
        useHarrisDetector=False,   # FalseならShi-Tomashi法
        # k=0.04,　　　　　　　　　# Harris法の測度に使用
    )
    '''

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
    print(first_frame.shape)


    #グレースケール変換
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # 読み込んだフレームの特徴点を探す
    prev_points = cv2.goodFeaturesToTrack(
        image=first_gray,      # 入力画像
        mask=None,            # mask=0のコーナーを無視
        **feature_params
    )

    # whileリープで読み込むための準備
    old_frame = first_frame
    old_gray = first_gray
    zahyou = prev_points.tolist() # 各フレームの特徴点の座標を入れておく
    for i in zahyou:
        i[0].append(0)


    print(zahyou)
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
            zahyou[rank].append([next_x, next_y, 0])
            
        
        # 次のフレームを読み込む準備
        old_gray = frame_gray.copy()
        prev_points = good_new.reshape(-1, 1, 2)
        result_img = frame
        framelist.append(1)
        
    return zahyou
    

if __name__=='__main__':
    import glob
    video_path = '/media/koshiba/Data/pix2pix/input/video'
    video_file = glob.glob(video_path + '/*')

    for path in video_file:
        todo(path)