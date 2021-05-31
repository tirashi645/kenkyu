def todo(path, savePath, mask):
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import math
    import datetime
    import pickle
    import glob
    import os
    import scipy.stats
    from PIL import Image
    from scipy import signal
    from pythonFile import k_means
    from pythonFile import normalization as normal
    from statistics import mode

    class1_err = []
    hz_fft2 = []
    num_fft2 = []
    hz_fft3 = []
    num_fft3 = []

    # 読み込む動画の設定
    videoName = path[path.rfind('/')+1:]
    cap = cv2.VideoCapture(path)
    print(videoName)

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
    zahyou = []
    all_zahyou = prev_points.tolist() # 各フレームの特徴点の座標を入れておく

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
        good_new = next_points[status == 1]
        good_old = prev_points[status == 1]
        
        # 対応点の是票を保存
        for rank, (prev_p, next_p) in enumerate(zip(good_old, good_new)):
            
            # x,y座標の取り出し
            # prev_x, prev_y: numpy.float32
            # next_x, next_y: numpy.float32
            prev_x, prev_y = prev_p.flatten()
            next_x, next_y = next_p.flatten()
            
            # 座標保存
            all_zahyou[rank].append([next_x, next_y])
            
        # 次のフレームを読み込む準備
        old_gray = frame_gray.copy()
        prev_points = good_new.reshape(-1, 1, 2)
        result_img = frame
        framelist.append(1)

    cap.release()

    # マスク内の特徴点のみをzahyouに入れる
    for i, data in enumerate(all_zahyou):
        if mask[i]==1:
            zahyou.append(data)
    print(np.array(zahyou).shape)
    print(np.array(all_zahyou).shape)

    # リストの中身を最小０,最大１に正規化する関数
    def min_max_normalization(x):
        x_min = min(x)
        x_max = max(x)
        x_norm = (x - x_min) / ( x_max - x_min)
        return x_norm

    #######################################################
    #Class1: 初期フレームとの誤差(x,y座標の直線距離)
    #######################################################
    def class1_output(k_err, zahyou):
        x_err = []
        y_err = []
        err = []
        max_err = 0
        min_err = 0
        max_index = -1
        min_index = -1
        # 特徴点ごとのx,y座標の絶対誤差をそれぞれ一つの配列にまとめる
        for i in range(len(k_err)):
            for j in range(len(k_err[0])):
                x_err.append(k_err[i][j][0])
                y_err.append(k_err[i][j][1])
        # 最小-1,最大1で正規化する
        x_err_normal = scipy.stats.zscore(x_err).tolist()
        y_err_normal = scipy.stats.zscore(y_err).tolist()
        # 正規化したx,y座標の絶対誤差のリストを一つのリストにまとめる
        for i in range(len(x_err)):
            err.append([x_err_normal[i], y_err_normal[i]])
            # リストのx,y座標の和が最大のものを抽出する
            if max_err < x_err_normal[i] + y_err_normal[i]:
                max_index = [x_err_normal[i], y_err_normal[i]]
            # リストのx,y座標の和が最小のものを抽出する
            elif min_err > x_err_normal[i] + y_err_normal[i]:
                min_index = [x_err_normal[i], y_err_normal[i]]
        #分類対象のデータのリスト。各要素はfloatのリスト
        vectors = err
        #分類対象のデータをクラスタ数3でクラスタリング
        centers = k_means.clustering_class1(vectors, 3, max_index, min_index)

        # 特徴点ごと実行:フレーム数分k-meansで分類して一番多い分類を採用する
        label = []
        k_err = scipy.stats.zscore(k_err).tolist()
        zahyou_ave = []

        # 分類したデータで各特徴点をクラスタリングする
        for frame in k_err:
            tmp = []
            sum_x = 0
            sum_y = 0
            # フレームごとにクラスタリングする
            for i in frame:
                tmp.append(k_means.near(i, centers))
                sum_x += i[0]
                sum_y += i[1]
            # クラスタリングした結果の最頻値をラベル付けする
            label_input = mode(tmp)
            # 絶対誤差の平均を計算する
            zahyou_ave.append([sum_x/len(frame), sum_y/len(frame)])
            # 0:noise→0   1,2:feature→1
            #if label_input == 2 or label_input == 1:
            #    label_input = 1
            #else:
            #    label_input = 0
            label.append(label_input) # 一番多い数字をlabelに追加

        x0 = []
        y0 = []
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        print(label)
        # 各特徴点の平均値をラベルに従いプロットする
        for index, zahyou_data in enumerate(zahyou_ave):
            if label[index]==0:
                x0.append(zahyou_data[0])
                y0.append(zahyou_data[1])
            elif label[index]==1:
                x1.append(zahyou_data[0])
                y1.append(zahyou_data[1])
            else:
                x2.append(zahyou_data[0])
                y2.append(zahyou_data[1])

        # figure
        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(1, 1, 1)

        # plot
        ax.scatter(x0, y0, color='r')
        ax.scatter(x1, y1, color='b')
        ax.scatter(x2, y2, color='g')

        #plt.title('Method-1', fontsize=36)
        plt.xlabel('victor in x', fontsize=36)
        plt.ylabel('victor in y', fontsize=36)
        plt.tick_params(labelsize=36)
        # プロットした画像を保存する
        plt.savefig(savePath + '/kmeans_class1_' + videoName[:-4] + '.jpg')

        return label


    #######################################################
    #Class2: 初期フレームとの誤差　(フーリエ変換)
    #######################################################
    def class2(data, hz_fft2, num_fft2):

        # 高速フーリエ変換
        F = np.fft.fft(data)

        F_abs = np.abs(F) # 複素数 ->絶対値に変換
        # 振幅を元の信号のスケールに揃える
        F_abs = F_abs / (N/2) # 交流成分
        F_abs[0] = F_abs[0] / 2 # 直流成分

        # FFTデータからピークを自動検出
        maximal_idx = signal.argrelmax(F_abs, order=1)[0] # ピーク（極大値）のインデックス取得

        # 閾値を確認
        tmp_fz = []
        
        # ピーク検出感度調整
        peak_cut = 0.0075 # ピーク閾値
        # 後半側（ナイキスト超）と閾値より小さい振幅ピークを除外
        maximal_idx = maximal_idx[(F_abs[maximal_idx] > peak_cut) & (maximal_idx <= N/2)]

        # 閾値以上のピーク数があるならば最大周波数とピーク数を取得
        if len(fq[maximal_idx])>0:
            hz_fft2.append(fq[maximal_idx][-1])
            num_fft2.append(len(fq[maximal_idx]))
        # 閾値以上のピーク数があるならば最大周波数とピーク数を０に設定
        else:
            hz_fft2.append(0)
            num_fft2.append(len(fq[maximal_idx]))

        return hz_fft2, num_fft2

    def class2_output(hz_fft2, num_fft2):
        max_sum = 0
        max_index = -1
        min_sum = 0
        min_index = -1
        fft = []
        # 最大周波数とピーク数のリストを一つのリストにまとめる
        for i in range(len(hz_fft2)):
            fft.append([hz_fft2[i], num_fft2[i]])
        # numpy配列に変換
        fft = np.array(fft)

        # 最小－1,最大1にリストを正規化
        normal_fft = scipy.stats.zscore(fft).tolist()
        # 正規化されたリストから最小の和と最大の和のリストを抽出
        for i in range(len(normal_fft)):
            sum_fft = normal_fft[i][0]+normal_fft[i][1]
            if max_sum < sum_fft:
                max_sum = sum_fft
                max_index = [normal_fft[i][0], normal_fft[i][1]]
            if min_sum > sum_fft:
                min_sum = sum_fft
                min_index = [normal_fft[i][0], normal_fft[i][1]]

        #分類対象のデータのリスト。各要素はfloatのリスト
        vectors = normal_fft
        #分類対象のデータをクラスタ数3でクラスタリング
        centers = k_means.clustering_class2(vectors, 3, max_index, min_index)

        label = []
        plot_label = []
        for i in normal_fft:
            label_input = (k_means.near(i, centers))
            plot_label.append(label_input)
            # 0,1:noise→0   2:feature→1
            #if label_input==0 or label_input==1:
            #    label.append(0)
            #else:
            #    label.append(1)
            label.append(label_input)

        # 各特徴点をラベルに従いプロットする
        fft_0x = []
        fft_0y = []
        fft_1x = []
        fft_1y = []
        fft_2x = []
        fft_2y = []
        print(label)
        for i in range(len(normal_fft)):
            if label[i]==0:
                fft_0x.append(normal_fft[i][0])
                fft_0y.append(normal_fft[i][1])
            elif label[i]==1:
                fft_1x.append(normal_fft[i][0])
                fft_1y.append(normal_fft[i][1])
            else:
                fft_2x.append(normal_fft[i][0])
                fft_2y.append(normal_fft[i][1])

        # figure
        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(1, 1, 1)

        # plot
        ax.scatter(fft_0x, fft_0y, color='b', s=36)
        ax.scatter(fft_1x, fft_1y, color='r', s=36)
        ax.scatter(fft_2x, fft_2y, color='g', s=36)

        #plt.title('Method-2', fontsize=36)
        plt.xlabel('vector in x', fontsize=36)
        plt.ylabel('vector in y', fontsize=36)
        plt.tick_params(labelsize=36)
        plt.savefig(savePath + '/kmeans_class2_' + videoName[:-4] + '.jpg')

        return label

    #######################################################
    #Class3: (初期フレーム － 1フレーム前)との誤差　(フーリエ変換)
    #######################################################
    def class3(data, hz_fft3, num_fft3):

        # 高速フーリエ変換
        F = np.fft.fft(data)

        F_abs = np.abs(F) # 複素数 ->絶対値に変換
        # 振幅を元の信号のスケールに揃える
        F_abs = F_abs / (N/2) # 交流成分
        F_abs[0] = F_abs[0] / 2 # 直流成分

        # FFTデータからピークを自動検出
        maximal_idx = signal.argrelmax(F_abs, order=1)[0] # ピーク（極大値）のインデックス取得
        # ピーク検出感度調整
        peak_cut = 0.01 # ピーク閾値
        # 後半側（ナイキスト超）と閾値より小さい振幅ピークを除外
        maximal_idx = maximal_idx[(F_abs[maximal_idx] > peak_cut) & (maximal_idx <= N/2)]

        # 閾値以上のピーク数があるならば最大周波数とピーク数を取得
        if len(fq[maximal_idx])>0:
            hz_fft3.append(fq[maximal_idx][-1])
            num_fft3.append(len(fq[maximal_idx]))
        # 閾値以上のピーク数があるならば最大周波数とピーク数を０に設定
        else:
            hz_fft3.append(0)
            num_fft3.append(len(fq[maximal_idx]))

        return hz_fft3, num_fft3

    def class3_output(hz_fft3, num_fft3):
        max_sum = 0
        max_index = -1
        min_sum = 0
        min_index = -1
        fft = []
        # 最大周波数とピーク数のリストを一つのリストにまとめる
        for i in range(len(hz_fft3)):
            fft.append([hz_fft3[i], num_fft3[i]])
        fft = np.array(fft)

        # 最小－1,最大1にリストを正規化
        normal_fft = scipy.stats.zscore(fft).tolist()
        # 正規化されたリストから最小の和と最大の和のリストを抽出
        for i in range(len(normal_fft)):
            sum_fft = normal_fft[i][0]+normal_fft[i][1]
            if max_sum < sum_fft:
                max_sum = sum_fft
                max_index = [normal_fft[i][0], normal_fft[i][1]]
            if min_sum > sum_fft:
                min_sum = sum_fft
                min_index = [normal_fft[i][0], normal_fft[i][1]]

        #分類対象のデータのリスト。各要素はfloatのリスト
        vectors = normal_fft
        #分類対象のデータをクラスタ数3でクラスタリング
        centers = k_means.clustering_class3(vectors, 3, max_index, min_index)

        plot_label = []
        label = []

        for i in normal_fft:
            label_input = (k_means.near(i, centers))
            plot_label.append(label_input)
            # ノイズではない2分類をまとめる
            if label_input==0 or label_input==1:
                label.append(0)
            else:
                label.append(1)
        
        # 各特徴点の平均値をラベルに従いプロットする
        fft_0x = []
        fft_0y = []
        fft_1x = []
        fft_1y = []
        fft_2x = []
        fft_2y = []
        print(label)
        for i in range(len(normal_fft)):
            if label[i]==0:
                fft_0x.append(normal_fft[i][0])
                fft_0y.append(normal_fft[i][1])
            elif label[i]==1:
                fft_1x.append(normal_fft[i][0])
                fft_1y.append(normal_fft[i][1])
            else:
                fft_2x.append(normal_fft[i][0])
                fft_2y.append(normal_fft[i][1])


        # figure
        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(1, 1, 1)

        # plot
        ax.scatter(fft_0x, fft_0y, color='g', s=36)
        ax.scatter(fft_1x, fft_1y, color='b', s=36)
        ax.scatter(fft_2x, fft_2y, color='r', s=36)

        plt.title('Method-3', fontsize=36)
        plt.xlabel('vector in x', fontsize=36)
        plt.ylabel('vector in y', fontsize=36)
        plt.tick_params(labelsize=36)
        plt.savefig(savePath + '/kmeans_class3_' + videoName[:-4] + '.jpg')

        return label


    '''
    主要な配列説明,要素数
    zahyou:特徴点x,y座標, (特徴点数,フレーム数,2)  :float
    vector1:フレーム間の座標誤差(ユークリッド距離), フレーム数-1  :float
    vector2:初期フレームとの座標誤差(ユークリッド距離), フレーム数-1  :float
    np_err: vector1 - vector2, フレーム数-1  :ndarray
    '''

    for point_data in zahyou:
        vector1 = []
        vector2 = []
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
        # 手法１で使うデータ
        class1_err.append(gosa)
        # 手法２で使うデータ
        vector_normal = min_max_normalization(vector1)
        np_vector_normal = np.array(vector_normal)
        # 手法３で使うデータ
        np_err = abs(np.array(vector2)-np.array(vector1))
        np_err_normal = min_max_normalization(np_err)

        # フーリエ変換に使う変数
        dt = 1.0/fps
        N = len(vector1)
        t = np.arange(0, N*dt, dt)
        fq = np.linspace(0,1.0/dt,N)

        # 手法２を実行
        hz_fft2, num_fft2 = class2(np_vector_normal, hz_fft2, num_fft2)

        # 手法３を実行
        hz_fft3, num_fft3 = class3(np_err_normal, hz_fft3, num_fft3)

    # 手法１～３を実行
    class1Data = class1_output(class1_err, zahyou)
    class2Data = class2_output(hz_fft2, num_fft2)
    class3Data = class3_output(hz_fft3, num_fft3)
    classList = [class1Data, class2Data, class3Data]
    return classList
    

if __name__=='__main__':
    from tkinter import filedialog
    from pythonFile import click_pct, timestump
    import glob
    import os
    # ファイルダイアログからファイル選択
    time = timestump.get_time()
    typ = [('','*')] 
    dir = 'C:\\pg'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 
    todo(path, time)