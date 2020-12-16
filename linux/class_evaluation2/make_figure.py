def todo(zahyou, savePath):
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle

    selectDir = ''
    c = ['b', 'r', 'g', 'y']
    skip_frame = 0     #最初の数フレームをスキップ

    f = open(savePath + '/category.txt', 'rb')
    catNum = pickle.load(f)

    for num, i in enumerate(zahyou):
        category = catNum[num]
        vec = []        # フレーム間のベクトルを入れる
        vec_first = []
        frame_num = []  # ベクトルとフレーム数を同期させる
        xList = []
        yList = []
        for frame, j in enumerate(i):            # フレーム間のベクトルを計算して配列に入れる
            if frame>=skip_frame:
                if frame==skip_frame:
                    first = np.array(j)
                if frame>skip_frame:
                    u = np.array(j) - a
                    vec.append(np.linalg.norm(u))
                    frame_num.append(frame)
                    xList.append(x - j[0])
                    yList.append(y - j[1])
                    u1 = np.array(j) - first
                    vec_first.append(np.linalg.norm(u1))
                a = np.array(j)
                x = j[0]
                y = j[1]

        # 各特徴点のカテゴリーごとに保存ディレクトリを分ける
        if category==0:
            #print(0)
            c = "b"
            selectDir = '/cat1/'
        elif category==1:
            #print(1)
            c = "r"
            selectDir = '/cat2/'
        elif category==2:
            #print(2)
            c = "g"
            selectDir = '/cat3/'
        elif category==3:
            #print(3)
            c = "y"
            selectDir = '/cat4/'
    
        fig_vec = plt.figure()
        ax = fig_vec.add_subplot()
        ax.plot(frame_num, vec, color=c)

        fig_first = plt.figure()
        ax3 = fig_first.add_subplot()
        ax3.plot(frame_num, vec_first, color=c)

        fig_error_x = plt.figure()
        ax1 = fig_error_x.add_subplot(211)
        ax2 = fig_error_x.add_subplot(212)
        ax1.plot([i for i in range(len(xList))], xList, 'r', label='x_error')
        ax2.plot([i for i in range(len(yList))], yList, 'b', label='y_error')

        handler1, label1 = ax1.get_legend_handles_labels()
        handler2, label2 = ax2.get_legend_handles_labels()

        ax1.legend(handler1, label1, loc=2, borderaxespad=0.)
        ax2.legend(handler2, label2, loc=2, borderaxespad=0.)

        fig_vec.savefig(savePath + selectDir + 'vec/' + str(num+1) + ".jpg")
        fig_first.savefig(savePath + selectDir + 'toFirst/' + str(num+1) + ".jpg")
        fig_error_x.savefig(savePath + selectDir + 'error/' + str(num+1) + ".jpg")
        print(savePath + selectDir + 'vec/' + str(num+1) + ".jpg")
        plt.close('all')

    print("Figure_finish")
    return 0

if __name__ == "__main__":
    from tkinter import filedialog
    import pickle
    import os
    import glob
    import shutil
    from pythonFile import make_dirs

    typ = [('','*')] 
    dir = '/media/koshiba/Data/opticalflow/point_data'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 
    videoDir = path[:path.rfind('/')]

    videolist = glob.glob(videoDir[:videoDir.rfind('/')] + "/*")
    print(videolist)

    print('Process all the files? (yes, no) :', end=" ")
    flag = input() 

    if flag == 'yes':
        for i in videolist:
            savepath = i
            print(savepath)
             
            videoName = i[i.rfind('/')+1:-4]

            dirName = videoDir[videoDir.rfind('/', 0, videoDir.rfind('/'))+1:]
            path = i + '/pointData_' + i[i.rfind('/')+1:] + '.txt'
            print(savepath, path)
            dirList = ['cat1', 'cat2', 'cat3', 'cat4']
            for d in dirList:
                make_dirs.deleteDir(savepath, '/' + d + '/vec')
                make_dirs.deleteDir(savepath, '/' + d + '/error')
            
            f = open(path, "rb")
            zahyou = pickle.load(f)

            a = todo(zahyou, savepath)
            
    else:
        savepath = path[:path.rfind('/')]
        print(savepath, path)
        videoName = path[path.rfind('/')+1:-4]
        dirList = ['cat1', 'cat2', 'cat3', 'cat4']
        for d in dirList:
            make_dirs.deleteDir(savepath, '/' + d + '/vec')
            make_dirs.deleteDir(savepath, '/' + d + '/error')

        f = open(path, "rb")
        zahyou = pickle.load(f)

        a = todo(zahyou, savepath)