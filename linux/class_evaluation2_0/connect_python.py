def doGet(path, videoName, savePath):
    import Make_wavedata, clusteringPoint, make_figure, make_fft
    import os
    import pickle
    import shutil

    print(path)

    #保存ディレクトリの作成
    if not os.path.isdir(savePath + '/cat1/pict'):
        for i in range(4):
            os.makedirs(savePath + '/cat' + str(i+1) + '/pict')
    if not os.path.isdir(savePath + '/fft'):
        os.makedirs(savePath + '/fft')
    else:
        shutil.rmtree(savePath + '/fft')
        os.makedirs(savePath + '/fft')

    zahyou = Make_wavedata.todo(path)   #オプティカルフローで各特徴点の移動を推定
    print(zahyou)
    zahyou = clusteringPoint.todo(path, zahyou) #手動で分類する
    make_figure.todo(zahyou, savePath, videoName)   #取得した特徴点の動きをグラフにする

    fft_savepaht = savePath + '/fft'
    make_fft.doGet(path, fft_savepaht, videoName)   #取得した特徴点の動きをFFT変換する
    

    f = open(savePath + '/pointData_' + videoName + '.txt', 'wb')
    pickle.dump(zahyou, f)
    
if __name__ == "__main__":
    from tkinter import filedialog
    import glob

    # ファイルダイアログからファイルを選択
    typ = [('','*')] 
    dir = 'C:\\pg'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)

    videoDir = path[:path.rfind('/')]
    videolist = glob.glob(videoDir + "/*")

    print('Process all the files? (yes, no) :', end=" ")
    flag = input() 

    if flag == 'yes':
        for i in videolist:
            dirName = videoDir[videoDir.rfind('/')+1:]
            videoName = i[i.rfind('\\')+1:]
            savePath = 'D:/opticalflow/point_data/' + dirName + '/' + videoName[:-4]
            videoPath = videoDir + '/' + videoName
            
            doGet(videoPath, videoName[:-4], savePath)
    else:
        dirName = videoDir[videoDir.rfind('/')+1:]
        videoName = path[path.rfind('/')+1:-4]
        savePath = 'D:/opticalflow/point_data/' + dirName + '/' + videoName
        videoPath = videoDir + '/' + videoName

        doGet(path, videoName, savePath)