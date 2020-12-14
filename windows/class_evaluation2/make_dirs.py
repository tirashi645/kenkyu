def makeDir(path):
    import os
    dirList = ['cat1', 'cat2', 'cat3', 'cat4', 'fft', 'seasonal']   # 第一階層のディレクトリ名
    inCatDir = ['vec', 'pict', 'error', 'toFirst', 'toFirstFFT']    # cat1~4内のディレクトリ名
    #ディレクトリを作成
    for index, dir_oya in enumerate(dirList):
        #print(path + '/' + dir_oya)
        if not os.path.isdir(path + '/' + dir_oya):
            print(path + '/' + dir_oya)
            os.makedirs(path + '/' + dir_oya)
        if index < 4:
            for dir_kodomo in inCatDir:
                if not os.path.isdir(path + '/' + dir_oya + '/' + dir_kodomo):
                    print(path + '/' + dir_oya + '/' + dir_kodomo)
                    os.makedirs(path + '/' + dir_oya + '/' + dir_kodomo)

def deleteDir(path, deletePath):
    import os
    import shutil

    if os.path.isdir(path + deletePath):
        print('delete:' + path + deletePath)
        shutil.rmtree(path + deletePath)
    makeDir(path)

if __name__ == "__main__":
    from tkinter import filedialog

    #ファイルを選択する
    typ = [('','*')] 
    dir = 'C:\\pg'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
    videoDir = path[:path.rfind('/')]
    dirName = videoDir[videoDir.rfind('/')+1:]
    videoName = path[path.rfind('/')+1:-4]
    dirPath = '/' + dirName + '/' + videoName
    savePath = 'D:/opticalflow/point_data' + dirPath
    print(savePath)
    makeDir(savePath)
    