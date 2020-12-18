def doGet(path):
    import pickle
    import numpy as np

    f = open(path, 'rb')
    zahyou = pickle.load(f)

    data = np.array([zahyou[i][0][2] for i in range(len(zahyou))])

    savepath = path[:path.rfind('/')]
    f = open(savepath + '/category.txt', 'wb')
    pickle.dump(data, f)


if __name__=='__main__':
    from tkinter import filedialog
    import glob
    import make_dirs, getVideoData

    typ = [('','*')] 
    dir = '/media/koshiba/Data/opticalflow/point_data'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
    videoPath = path[:path.rfind('/')]
        
    videolist = glob.glob(videoPath[:videoPath.rfind('/')] + "/*")

    print('Process all the files? (yes, no) :', end=" ")
    flag = input() 

    if flag == 'yes':
        for i in videolist:
            savepath = i
            dirName = videoPath[videoPath.rfind('/', 0, videoPath.rfind('/'))+1:]
            path = i + '/pointData_' + i[i.rfind('/')+1:] + '.txt'
            print(path)

            doGet(path)
    else:
        savepath = path[:path.rfind('/')]
        dirName = videoPath[videoPath.rfind('/', 0, videoPath.rfind('/'))+1:]
        print(path)

        doGet(path)
    print('finish')