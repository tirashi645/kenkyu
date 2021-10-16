from numpy.lib.npyio import save


def doGet(path, videoName, savePath):
    import Make_wavedata, clusteringPoint, make_figure, make_fft
    from pythonFile import make_dirs, getVideoData
    import os
    import pickle
    import shutil

    print(path, videoName, savePath)

    dirName = getVideoData.getDirName(path)
    videoName = getVideoData.getVideoName(path)
    #保存ディレクトリの作成
    make_dirs.makeDir(savePath)

    zahyou = Make_wavedata.todo(path)   #オプティカルフローで各特徴点の移動を推定
    print(zahyou)
    if os.path.isdir('/media/koshiba/Data/opticalflow/point_data/' + dirName + '/' + videoName + '/category.txt'):
        f = open('/media/koshiba/Data/opticalflow/point_data/' + dirName + '/' + videoName + '/category.txt', 'rb')
        noise = pickle.load(f)
    else:
        noise = clusteringPoint.todo(path, zahyou) #手動で分類する
    make_figure.todo(zahyou, savePath)   #取得した特徴点の動きをグラフにする
    
    make_fft.doGet(zahyou, savePath)   #取得した特徴点の動きをFFT変換する
        

    predList = [[],[],[]]
    accuracy = ['-1', '-1', '-1']
    precision = ['-1', '-1', '-1']
    recall = ['-1', '-1', '-1']
    specificity = ['-1', '-1', '-1']
    tmp = 0

    for index1, pred in enumerate(classList):
        for index2, answer in enumerate(noise):
            #print(index1, index2)
            if (pred[index2]==0 or pred[index2]==-1):
                if answer==0:
                    predList[index1].append(0)
                else:
                    predList[index1].append(3)
            else:
                if answer==1:
                    predList[index1].append(1)
                else:
                    predList[index1].append(2)
        
        predAll = len(predList[index1])
        tp = predList[index1].count(1)
        tn = predList[index1].count(0)
        fp = predList[index1].count(2)
        fn = predList[index1].count(3)

        print(predAll, tp, tn)
        accuracy[index1] = str((tp + tn)/predAll)
        if tp+fp != 0:
            precision[index1] = str(tp/(tp+fp))
        if tp+fn != 0:
            recall[index1] = str(tp/(tp+fn))
        if tn+fp != 0:
            specificity[index1] = str(tn/(fp+tn))

    print(classList)

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    print('accuracy:' + accuracy[0] + ' ' + accuracy[1] + ' '  + accuracy[2])
    print('precision' + precision[0] + ' '  + precision[1] + ' '  + precision[2])
    print('recall' + recall[0] + ' '  + recall[1] + ' '  + recall[2])
    print('specificity' + specificity[0] + ' '  + specificity[1] + ' '  + specificity[2])

    f = open(savePath + '/pointData_' + videoName + '.txt', 'wb')
    pickle.dump(zahyou, f)
    

    
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
            savePath = '/media/koshiba/Data/opticalflow/point_data/' + dirName + '/' + videoName[:-4]
            print(savePath)
            videoPath = videoDir + '/' + videoName
            
            doGet(videoPath, videoName[:-4], savePath)
    else:
        dirName = videoDir[videoDir.rfind('/')+1:]
        videoName = path[path.rfind('/')+1:-4]
        savePath = '/media/koshiba/Data/opticalflow/point_data/' + dirName + '/' + videoName
        print(savePath)
        videoPath = videoDir + '/' + videoName

        doGet(path, videoName, savePath)