from numpy.lib.npyio import save


def todo(path):
    from pythonFile import Make_wavedata, make_figure, make_fft, make_dirs
    import video_proc
    import os
    import pickle
    import shutil
    import cv2

    videoName = path.split('/')[-1][:-4]

    savePath = '/media/koshiba/Data/pix2pix/output/proc_point/' + videoName
    print(path, videoName, savePath)

    #保存ディレクトリの作成
    #make_dirs.makeDir(savePath)

    # 動画の設定
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_setting = [width, height, fps]

    with open('/media/koshiba/Data/pix2pix/input/point/' + videoName + '.pickle', 'rb') as f:
        noise = pickle.load(f)

    mask_data = video_proc.todo(path)
    mask_point = mask_data[1]

    classList = Make_wavedata.todo(path, savePath, mask_point)   #オプティカルフローで各特徴点の移動を推定
    
    #make_figure.todo(zahyou, savePath, video_setting, mask_point)   #取得した特徴点の動きをグラフにする
    
    #make_fft.doGet(zahyou, savePath, mask_point)   #取得した特徴点の動きをFFT変換する

    class_evalute = []

    for num, class_point in enumerate(classList):
        point_data = mask_point
        point_evalute = []
        cnt = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        print(mask_point)

        for i, data in enumerate(mask_point):
            if data:
                point_data[i] = class_point[cnt]
                cnt += 1
        classList[num] = point_data
        for i in range(len(point_data)):
            if point_data[i]==1 and noise[i]==1:
                tp+=1
                point_evalute.append('tp')
            elif point_data[i]==1 and noise[i]==0:
                fn+=1
                point_evalute.append('fn')
            elif point_data[i]==0 and noise[i]==0:
                tn+=1
                point_evalute.append('tn')
            elif point_data[i]==0 and noise[i]==1:
                fp+=1
                point_evalute.append('fp')

        print('[tp, tn, fp, fn]')
        print(tp, tn, fp, fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        specificity = tn/(fp+tn)
        f_value = (2 * recall * precision) / (recall + precision)

        print('acc:{:.3f}, pre:{:.3f}, rec:{:.3f}, spe:{:.3f}, f_value:{:.3f}'.format(accuracy, precision, recall, specificity, f_value))
        class_evalute.append([accuracy, precision, recall, specificity, f_value])

    return class_evalute
    

    
if __name__ == "__main__":
    import glob
    import os
    import pickle

    # ファイルダイアログからファイル選択
    '''
    typ = [('','*')] 
    dir = 'C:\\pg'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 
    '''
    video_path = '/media/koshiba/Data/pix2pix/input/video'
    video_file = glob.glob(video_path + '/*')

    accuracy = [0, 0, 0]
    precision = [0, 0, 0]
    recall = [0, 0, 0]
    specificity = [0, 0, 0]
    f_value = [0, 0, 0]

    acc_list = [[-1.0, 10000.0], [-1.0, 10000.0], [-1.0, 10000.0]]
    f_list = [[-1.0, 10000.0], [-1.0, 10000.0], [-1.0, 10000.0]]
    max_acc = [[], [], []]
    min_acc = [[], [], []]
    max_fValue = [[], [], []]
    min_fValue = [[], [], []]
    minmax = [[], [], []]
    value_list = [[], [], []]

    for path in video_file:

        class_list = todo(path)
        videoName = path.split('/')[-1][:-4]


        #######################################################
        # ここから評価計算
        #######################################################
        for i, value_list in enumerate(class_list):
            accuracy[i] += value_list[0]
            precision[i] += value_list[1]
            recall[i] += value_list[2]
            specificity[i] += value_list[3]
            f_value[i] += value_list[4]
            if acc_list[i][0] < value_list[0]:
                max_acc[i] = [videoName, value_list]
                acc_list[i][0] = value_list[0]
            if acc_list[i][1] > value_list[0]:
                min_acc[i] = [videoName, value_list]
                acc_list[i][1] = value_list[0]

            if f_list[i][0] < value_list[4]:
                max_fValue[i] = [videoName, value_list]
                f_list[i][0] = value_list[4]
            if f_list[i][1] > value_list[4]:
                min_fValue[i] = [videoName, value_list]
                f_list[i][1] = value_list[4]
        for i in range(3):
            accuracy[i] /= len(video_file)
            precision[i] /= len(video_file)
            recall[i] /= len(video_file)
            specificity[i] /= len(video_file)
            f_value[i] /= len(video_file)

            value_list[i] = [accuracy[i], precision[i], recall[i], specificity[i], f_value[i]]
            minmax[i] = [max_acc[i], min_acc[i], max_fValue[i], min_fValue[i]]
            
           
    with open('/media/koshiba/Data/pix2pix/output/proc_point/evalute.pickle', 'wb') as f:
        pickle.dump(video_file, f)
    with open('/media/koshiba/Data/pix2pix/output/proc_point/minmax.pickle', 'wb') as f:
        pickle.dump(minmax, f)

    for i in range(3):
        print('class' + str(i) + '------------------------')
        print('acc:{:.3f}, pre:{:.3f}, rec:{:.3f}, spe:{:.3f}, f_value:{:.3f}'.format(accuracy[i], precision[i], recall[i], specificity[i], f_value[i]))
        print(acc_list[i],f_list[i])
        print('max_acc-----------------------')
        print(max_acc[i])
        print('min_acc-----------------------')
        print(min_acc[i])
        print('max_fValue--------------------')
        print(max_fValue[i])
        print('min_fValue--------------------')
        print(min_fValue[i])
