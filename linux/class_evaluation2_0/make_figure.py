def todo(zahyou, savePath, videoName):
    print(zahyou)
    import matplotlib.pyplot as plt
    import numpy as np

    selectDir = ''
    c = ['b', 'r', 'g', 'y']
    skip_frame = 10     #最初の数フレームをスキップ

    for num, i in enumerate(zahyou):
        vec = []        # フレーム間のベクトルを入れる
        frame_num = []  # ベクトルとフレーム数を同期させる
        xList = []
        yList = []
        for frame, j in enumerate(i):            # フレーム間のベクトルを計算して配列に入れる
            if frame>=skip_frame:
                if frame>skip_frame:
                    u = np.array(j) - a
                    vec.append(np.linalg.norm(u))
                    frame_num.append(frame)
                    xList.append(x - j[0])
                    yList.append(y - j[1])
                a = np.array(j)
                x = j[0]
                y = j[1]

        # 各特徴点のカテゴリーごとに保存ディレクトリを分ける
        if i[0][2]==0:
            print(0)
            c = "b"
            selectDir = '/cat1/'
        elif i[0][2]==1:
            print(1)
            c = "r"
            selectDir = '/cat2/'
        elif i[0][2]==2:
            print(2)
            c = "g"
            selectDir = '/cat3/'
        elif i[0][2]==3:
            print(3)
            c = "y"
            selectDir = '/cat4/'
    
        fig_vec = plt.figure()
        ax = fig_vec.add_subplot()
        ax.plot(frame_num, vec, color=c)

        fig_error = plt.figure()
        #fig_error_y = plt.figure()
        ax1 = fig_error.add_subplot()
        ax2 = ax1.twinx()
        ax1.plot([i for i in range(len(xList))], xList, 'r', label='x_error')
        ax2.plot([i for i in range(len(yList))], yList, 'b', label='y_error')

        handler1, label1 = ax1.get_legend_handles_labels()
        handler2, label2 = ax2.get_legend_handles_labels()

        ax1.legend(handler1 + handler2, label1 + label2, loc=2, borderaxespad=0.)

        fig_vec.savefig(savePath + selectDir + '/vec/' + videoName + '_' + str(num+1) + ".jpg")
        fig_error.savefig(savePath + selectDir + '/error/' + videoName + '_' + str(num+1) + ".jpg")
        print(savePath + selectDir + '/vec/' + videoName + '_' + str(num+1) + ".jpg")
        plt.close()
        plt.close()

    return 0

if __name__ == "__main__":
    from tkinter import filedialog
    import pickle
    import os
    import glob
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
            print(savepath)
             
            videoName = i[i.rfind('\\')+1:-4]

            dirName = videoDir[videoDir.rfind('/', 0, videoDir.rfind('/'))+1:]
            path = i + '/pointData_' + i[i.rfind('\\')+1:] + '.txt'
            print(savepath, path)
            if not os.path.isdir(savepath + '/cat4/error'):
                for num in range(4):
                    os.makedirs(savepath + '/cat' + str(num+1) + '/vec')
                    os.makedirs(savepath + '/cat' + str(num+1) + '/error')
            else:
                for num in range(4):
                    tmpList = glob.glob(savepath + '/cat' + str(num+1) + '/*.jpg')
                    pictData = savepath + '/cat' + str(num+1) + '/*_cat' + str(num+1) + '/.jpg'
                    for fileName in tmpList:
                        if fileName != pictData:
                            os.remove(fileName)
                    shutil.rmtree(savepath + '/cat' + str(num+1) + '/vec')
                    shutil.rmtree(savepath + '/cat' + str(num+1) + '/error')
            f = open(path, "rb")
            zahyou = pickle.load(f)

            a = todo(zahyou, savepath, videoName)
            
    else:
        savepath = path[:path.rfind('/')]
        print(savepath, path)
        videoName = path[path.rfind('/')+1:-4]

        if not os.path.isdir(savepath + '/cat4/error'):
            for num in range(4):
                os.makedirs(savepath + '/cat' + str(num+1) + '/vec')
                os.makedirs(savepath + '/cat' + str(num+1) + '/error')
        else:
            for num in range(4):
                tmpList = glob.glob(savepath + '/cat' + str(num+1) + '/*.jpg')
                pictData = savepath + '/cat' + str(num+1) + '/*_cat' + str(num+1) + '/.jpg'
                for fileName in tmpList:
                    if fileName != pictData:
                        os.remove(fileName)
                shutil.rmtree(savepath + '/cat' + str(num+1) + '/vec')
                shutil.rmtree(savepath + '/cat' + str(num+1) + '/error')
                os.makedirs(savepath + '/cat' + str(num+1) + '/vec')
                os.makedirs(savepath + '/cat' + str(num+1) + '/error')
        f = open(path, "rb")
        zahyou = pickle.load(f)

        a = todo(zahyou, savepath, videoName)