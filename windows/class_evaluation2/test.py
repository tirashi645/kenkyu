if __name__ == "__main__":
    from tkinter import filedialog
    import glob
    import make_dirs

    typ = [('','*')] 
    dir = 'C:\\pg'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
    videoDir = path[:path.rfind('/')]
    dirName = videoDir[videoDir.rfind('/')+1:]
    videoName = path[path.rfind('/')+1:-4]
    dirPath = '/' + dirName + '/' + videoName
    print(setPath + dirPath)
    make_dirs.todo(dirPath)
    