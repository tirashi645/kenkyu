def doGet(zahyou, savePath):
    import pickle
    import numpy as np
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt
    import pickle
    import pandas

    skip_frame = 0

    # ビデオの情報を取得
    width = 730
    height = 1290
    fps = 30

    f = open(savePath + '/category.txt', 'rb')
    catNum = pickle.load(f)

    for num, i in enumerate(zahyou):
        category = catNum[num]
        vec = []
        firstList_x = []
        firstList_y = []
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
                    firstList_x.append(first_x - j[0])
                    firstList_y.append(first_y - j[1])
                    frame_num.append(frame)
                elif frame==skip_frame:
                    first_x = j[0]
                    first_y = j[1]
                a = np.array(j)
        if flag==0:
            #外れ値の削除
            #print(vec)
            #vec = hazurechi(vec)
            doFFT(vec, num+1, category, fps, frame_num, savePath)
            doFirstFFT(firstList_x, firstList_y, num+1, category, fps, frame_num, savePath)
    print('FFT_finish')

def doFFT(np_vector, num, category, fps, frame_num, savePath):
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
    F_abs[0] = 0

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

    plt.savefig(savePath + '/cat' + str(category + 1) + '/toFirstFFT/' + str(num))

    plt.close()

    #getSeasonal(np_vector, fq, savePath, num)

def doFirstFFT(xList, yList, num, category, fps, frame_num, savePath):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    
    # 必要な変数の準備
    dt = 1.0/fps
    N = len(xList)
    t = np.arange(0, N*dt, dt)
    fq = np.linspace(0,1.0/dt,N)
    xF = np.fft.fft(xList)
    yF = np.fft.fft(yList)

    xF_abs = np.abs(xF)
    xF_abs = xF_abs / (N/2)
    xF_abs[0] = 0

    yF_abs = np.abs(yF)
    yF_abs = yF_abs / (N/2)
    yF_abs[0] = 0

    fig_error = plt.figure(figsize=(40,12))
    error_x = fig_error.add_subplot(221)
    error_y = fig_error.add_subplot(223)
    fft_x = fig_error.add_subplot(222)
    fft_y = fig_error.add_subplot(224)

    error_x.plot([i for i in range(len(xList))], xList, 'r', linewidth=3, label='x_error')
    error_y.plot([i for i in range(len(yList))], yList, 'b', linewidth=3, label='y_error')

    fft_x.axis([0,1.0/dt/2,0,max(xF_abs)*1.5])
    fft_x.plot(fq, xF_abs, linewidth=3)
    
    fft_y.axis([0,1.0/dt/2,0,max(yF_abs)*1.5])
    fft_y.plot(fq, yF_abs, linewidth=3)
    error_x.tick_params(labelsize=18)
    error_y.tick_params(labelsize=18)
    fft_x.tick_params(labelsize=18)
    fft_y.tick_params(labelsize=18)

    plt.savefig(savePath + '/cat' + str(category + 1) + '/toFirstErrorFFT/' + str(num))
    print(savePath + '/cat' + str(category + 1) + '/toFirstErrorFFT/' + str(num) + ".jpg")
    plt.close('all')

# getSeasonal()は未実装
def getSeasonal(test_data, fq, savePath, num):
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
    plt.savefig(savePath + '\\seasonal\\' + str(num))
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
    from pythonFile import make_dirs, getVideoData
    import pickle

    #path:pointData_"VideoName".txt
    #savePath:~/opticalflow/point_data/"DirName"/"VideoName"

    typ = [('','*')] 
    dir = '/media/koshiba/Data/opticalflow/point_data'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
    videoPath = path[:path.rfind('/')]
        
    videolist = glob.glob(videoPath[:videoPath.rfind('/')] + "/*")

    print('Process all the files? (yes, no) :', end=" ")
    flag = input() 

    print("file name: ", end="")
    videoName = input()

    color = ['red','blue','green']

    if flag == 'yes':
        for i in color:
            savePath = '/media/koshiba/Data/opticalflow/point_data/sepa_color/' + i + '/' + videoName
            path = savePath + '/pointData_' + videoName + '.txt'
            print(savePath, path)
            make_dirs.deleteDir(savePath, '/fft')
            make_dirs.deleteCatDir(savePath, '/toFirstErrorFFT')

            f = open(path, "rb")
            zahyou = pickle.load(f)

            doGet(zahyou, savePath)
    else:
        savePath = path[:path.rfind('/')]
        dirName = videoPath[videoPath.rfind('/', 0, videoPath.rfind('/'))+1:]
        print(savePath, path)
        make_dirs.deleteDir(savePath, '/fft')
        make_dirs.deleteCatDir(savePath, '/toFirstErrorFFT')
            
        f = open(path, "rb")
        zahyou = pickle.load(f)

        doGet(zahyou, savePath)
