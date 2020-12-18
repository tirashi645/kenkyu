def getVec(zahyou):
    import numpy as np

    # 特徴点データからベクトルを取得（フレーム間）
    vecList = []
    for point in zahyou:
        vec = []
        for num, frame in enumerate(point):
            if  num!=0:
                u = np.array(frame) - a
                vec.append(np.linalg.norm(u))   # ユークリッド距離ベクトルを計算
            a = np.array(frame)
        vecList.append(vec)
    return vecList

def todo(path):
    import pickle
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    txtName = path[path.rfind('/')+1:]  # 特徴点データのファイル名
    dirPath = path[:path.rfind('/')]    # 特徴点データのディレクトリ名
    winsizeList = [num for num in range(5,41)]  # windowSizeの範囲
    videoName = dirPath[dirPath.rfind('/')+1:]
    print(dirPath, txtName)
    maxList = []    # 各特徴点の各windowSizeの最大値を取得
    aveList = []    # windowSizeごとの特徴点の平均を取得
    for i in range(5,41):
        # 特徴点データを読み込む
        txtPath = dirPath + '/winsize/winsize_' + str(i) + '/' + txtName
        f = open(txtPath, "rb")
        zahyou = pickle.load(f)

        vecList = getVec(zahyou)    # すべての特徴点のフレーム間ベクトルを取得
        tmp_max = []
        for vec in vecList:
            tmp_max.append(max(vec))    # 各特徴点のベクトルの最大値を取得
        maxList.append(tmp_max)
        aveList.append(sum(tmp_max)/len(tmp_max))
    maxList = np.transpose(np.array(maxList))   # listの次元を入れ替える

    # 保存先のディレクトリがなければ作成
    if not os.path.isdir(dirPath + '/winsize/graph'):
        os.makedirs(dirPath + '/winsize/graph')

    # 各特徴点ごとにwindowSizeの最大値を折れ線グラフで出力  X軸：windowSize Y軸:ベクトルの最大値
    for i in range(len(zahyou)):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(winsizeList, maxList[i])

        fig.savefig(dirPath + '/winsize/graph/' + str(i) + '.jpg')
        plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(title=videoName+'_average')
    ax.plot(winsizeList, aveList)   # 各特徴点のベクトル最大値の平均を折れ線グラフで出力    X軸：windowSize Y軸:ベクトルの最大値
    fig.savefig(dirPath + '/winsize/average.jpg')

    # グラフのY軸上限を25,50,75,100に設定して出力
    for num in range(4):
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(title=videoName+'_average' ,ylim = (0, (num+1)*25))
        ax2.plot(winsizeList, aveList)
        fig2.savefig(dirPath + '/winsize/average' + str((num+1)*25) + '.jpg')

    plt.close('all')

if __name__ == "__main__":
    from tkinter import filedialog
    import glob
    import os

    typ = [('','*')] 
    if os.path.isdir('/media/koshiba/Data/opticalflow/point_data'):
        dir = '/media/koshiba/Data/opticalflow/point_data'
    else:
        dir = 'C:\\pg'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
    videoDir = path[:path.rfind('/')]

    videolist = glob.glob(videoDir[:videoDir.rfind('/')] + "/**/pointData_*.txt")
    
    print('Process all the files? (yes, no) :', end=" ")
    flag = input() 

    if flag=="yes":
        for path in videolist:
            videoDir = path[:path.rfind('/')] + '/winsize'
            if os.path.isdir(videoDir):
                todo(path)

    else:
        videoDir = path[:path.rfind('/')]
        if os.path.isdir(videoDir):
            todo(path)
    
    print('finish')