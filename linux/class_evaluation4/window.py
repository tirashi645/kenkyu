if __name__=='__main__':
    import Make_wavedata, clusteringPoint, make_figure, make_fft
    from pythonFile import make_dirs, getVideoData
    import os
    import pickle
    import shutil
    from tkinter import filedialog
    import glob

    # ファイルをGUIで選択してPATHを取得
    typ = [('','*')] 
    dir = '/media/koshiba/Data/opticalflow/point_data'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
    videoPath = path[:path.rfind('/')]  # 解析するビデオ名

    error = []
        
    videolist = glob.glob(videoPath[:videoPath.rfind('/')] + "/*/")     # 解析済みの動画のディレクトリをすべて取得
    print(videolist)

    print('Process all the files? (yes, no) :', end=" ")
    flag = input() 

    if flag == 'yes':   # すべての動画で実行する
        for i in videolist:
            # 動画名とディレクトリ名を取得
            dirName = getVideoData.getDirName(i)
            videoName = getVideoData.getVideoName(i)

            videoPath = '/media/koshiba/Data/video/yohaku/' + dirName + '/' + videoName + '.avi'
            print(videoPath)

            # 選択された動画の解析が終わっていれば実行
            if os.path.exists(videoPath):
                for num in range(5,41):     # windowSizeを5~40で実行
                    if not os.path.isdir(i + 'window_' + str(num) + '/cat4/toFirstErrorFFT'):
                        make_dirs.makeDir(i[:-1] + '/winsize/winsize_' + str(num))  # 保存するフォルダがなければ作成
                    savePath = i + 'winsize/winsize_' + str(num)    # 保存先のPATH
                    zahyou = Make_wavedata.todo(videoPath, num)     # 動画を解析する（時間はかかるけど毎回動画を読み込む），特徴点の各フレームにおける座標を取得
                    make_figure.todo(zahyou, savePath)              # 各特徴点の各波形(vec, error, toFirstError)を取得
                    make_fft.doGet(zahyou, savePath)                # 各波形のFFT解析(toFirstFFT, toFirstErrorFFT)
            else:
                error.append(videoName)
        print("Could not be executed:" + error)

    else:
        dirName = getVideoData.getDirName(path)
        videoName = getVideoData.getVideoName(path)

        videoPath = '/media/koshiba/Data/video/yohaku/' + dirName + '/' + videoName + '.avi'

        for num in range(5,7):
            if not os.path.isdir(path[:path.rfind('/')] + '/winsize/winsize_' + str(num) + '/cat4/toFirstErrorFFT'):
                make_dirs.makeDir(path[:path.rfind('/')] + '/winsize/winsize_' + str(num))
            savePath = path[:path.rfind('/')] + '/winsize/winsize_' + str(num)
            Make_wavedata.todo(videoPath, num)
            zahyou = Make_wavedata.todo(videoPath, num)
            make_figure.todo(zahyou, savePath)
            make_fft.doGet(zahyou, savePath)

