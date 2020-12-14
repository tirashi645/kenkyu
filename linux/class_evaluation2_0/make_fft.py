def doGet(path, savepath, dirName):
    import pickle
    import numpy as np
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt
    import pickle
    import pandas

    # ビデオの準備
    path2 = 'D:\opticalflow\\video\\' + dirName + '.avi'
    cap = cv2.VideoCapture(path2)
    skip_frame = 10

    # ビデオの情報を取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 特徴点データを読み込み(Make_wavedata.pyで取得したもの)
    f = open(path, 'rb')
    zahyou = pickle.load(f)

    for num, i in enumerate(zahyou):
        category = i[0][2]
        vec = []
        frame_num = []
        flag = 0
        for frame, j in enumerate(i):
            # 特徴点がビデオのふちに（10ピクセル）いった場合に特徴点の移動を中止する
            if j[0]<10 or j[0]>width-10 or j[1]<10 or j[1]>height-10:
                if frame <= 10:
                    flag = 1
                    break
                else:
                    break
            if frame>=skip_frame:
                if frame>skip_frame:
                    u = np.array(j) - a
                    vec.append(np.linalg.norm(u))
                    frame_num.append(frame)
                a = np.array(j)
        if flag==0:
            #外れ値の削除
            print(vec)
            vec = hazurechi(vec)
            doFFT(vec, num+1, category, fps, frame_num, savepath)
    print('finish')

def doFFT(np_vector, num, category, fps, frame_num, savepath):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    
    # 必要な変数の準備
    dt = 1.0/fps
    N = len(np_vector)
    t = np.arange(0, N*dt, dt)
    fq = np.linspace(0,1.0/dt,N)
    F = np.fft.fft(np_vector)

    F_abs = np.abs(F)
    F_abs = F_abs / (N/2)
    F_abs[0] = F_abs[0] / 2

    plt.figure(figsize=(20,6))

    maximal_idx = signal.argrelmax(F_abs, order=1)[0]
    peak_cut = 0.011
    maximal_idx = maximal_idx[(F_abs[maximal_idx] > peak_cut) & (maximal_idx <= N/2)]

    plt.subplot(122)
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Amplitude')

    plt.axis([0,1.0/dt/2,0,max(F_abs)*1.5])
    #plt.ylim(0, 0.1)
    plt.plot(fq, F_abs)
    #plt.plot(fq[maximal_idx], F_abs[maximal_idx],'ro')
    
    plt.subplot(121)
    plt.plot([i for i in range(len(np_vector))], np_vector)
    plt.title('cat' + str(category + 1))
    
    plt.savefig(savepath + '\\fft\\cat' + str(category + 1) + '_' + str(num))
    plt.close()

    #getSeasonal(np_vector, fq, savepath, num)

# getSeasonal()は未実装
def getSeasonal(test_data, fq, savepath, num):
    import statsmodels.api as sm
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    test_data = pd.Series(test_data, index=[i+1 for i in range(len(test_data))])
    res = sm.tsa.seasonal_decompose(test_data, fq)

    original = test_data # オリジナルデータ
    trend = res.trend.dropna(how='all') # トレンドデータ
    seasonal = res.seasonal # 季節性データ
    residual = res.resid # 残差データ
    sr = seasonal + trend

    plt.figure(figsize=(8, 8)) # グラフ描画枠作成、サイズ指定

    # オリジナルデータのプロット
    plt.subplot(511) # グラフ4行1列の1番目の位置（一番上）
    plt.plot(original)
    plt.ylabel('Original')

    # trend データのプロット
    plt.subplot(512) # グラフ4行1列の2番目の位置
    plt.plot(trend)
    plt.ylabel('Trend')

    # seasonalデータ のプロット
    plt.subplot(513) # グラフ4行1列の3番目の位置
    plt.plot(seasonal)
    plt.ylabel('Seasonality')

    # residual データのプロット
    plt.subplot(514) # グラフ4行1列の4番目の位置（一番下）
    plt.plot(residual)
    plt.ylabel('Residuals')

    # residual データのプロット
    plt.subplot(515) # グラフ4行1列の4番目の位置（一番下）
    plt.plot(sr)
    plt.ylabel('trend + residual')

    plt.tight_layout() # グラフの間隔を自動調整
    plt.savefig(savepath + '\\seasonal\\' + str(num))
    plt.close()

def hazurechi(data):
    import numpy as np
    m = np.mean(data)
    s = np.std(data)
    while(True):
        vec = [e for e in data if (m - 1.5*s < e < m + 1.5*s)]
        if len(vec)==len(data):
            break
        data = vec
    return vec

if __name__ == "__main__":
    from tkinter import filedialog
    import glob
    import os
    import shutil

    typ = [('','*')] 
    dir = 'C:\\pg'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
    videoDir = path[:path.rfind('/')]
        
    videolist = glob.glob(videoDir[:videoDir.rfind('/')] + "/*")

    print('Process all the files? (yes, no) :', end=" ")
    flag = input() 

    if flag == 'yes':
        for i in videolist:
            savepath = i
            dirName = videoDir[videoDir.rfind('/', 0, videoDir.rfind('/'))+1:]
            path = i + '/pointData_' + i[i.rfind('\\')+1:] + '.txt'
            print(savepath, path)
            if not os.path.isdir(savepath + '/fft'):
                os.makedirs(savepath + '/fft')
            else:
                shutil.rmtree(savepath + '/fft')
                os.makedirs(savepath + '/fft')
            if not os.path.isdir(savepath + '/seasonal'):
                os.makedirs(savepath + '/seasonal')
            
            doGet(path, savepath, dirName)
    else:
        savepath = path[:path.rfind('/')]
        dirName = videoDir[videoDir.rfind('/', 0, videoDir.rfind('/'))+1:]
        print(savepath, path)

        if not os.path.isdir(savepath + '/fft'):
            os.makedirs(savepath + '/fft')
        else:
            shutil.rmtree(savepath + '/fft')
            os.makedirs(savepath + '/fft')
        if not os.path.isdir(savepath + '/seasonal'):
            os.makedirs(savepath + '/seasonal')
            
        doGet(path, savepath, dirName)
