def doGet(path, videoName, savePath):
    import Make_wavedata, clusteringPoint, make_figure, make_fft
    from pythonFile import make_dirs
    import os
    import pickle
    import shutil

    print(path, videoName, savePath)
    #保存ディレクトリの作成
    make_dirs.makeDir(savePath)

    zahyou = Make_wavedata.todo(path)   #オプティカルフローで各特徴点の移動を推定
    print(zahyou)
    zahyou = clusteringPoint.todo(path, zahyou) #手動で分類する
    make_figure.todo(zahyou, savePath)   #取得した特徴点の動きをグラフにする
    
    make_fft.doGet(path, savePath)   #取得した特徴点の動きをFFT変換する
    

    
if __name__ == "__main__":
    from tkinter import filedialog
    import glob

    # ファイルダイアログからファイルを選択
    typ = [('','*')] 
    dir = '/media/koshiba/Data/video'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)

    videoDir = path[:path.rfind('/')]
    videolist = glob.glob(videoDir + "/*")
    print(path, videoDir)

    print('Process all the files? (yes, no) :', end=" ")
    flag = input() 

    if flag == 'yes':
        for i in videolist:
            dirName = videoDir[videoDir.rfind('/')+1:]
            videoName = i[i.rfind('/')+1:]
            savePath = 'D:/opticalflow/point_data/' + dirName + '/' + videoName[:-4]
            videoPath = videoDir + '/' + videoName
            
            doGet(videoPath, videoName[:-4], savePath)
    else:
        dirName = videoDir[videoDir.rfind('/')+1:]
        videoName = path[path.rfind('/')+1:-4]
        savePath = 'D:/opticalflow/point_data/' + dirName + '/' + videoName
        videoPath = videoDir + '/' + videoName

        doGet(path, videoName, savePath)