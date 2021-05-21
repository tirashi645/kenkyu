
import cv2
import numpy as np

class mouseParam:
    def __init__(self, input_img_name):
        #マウス入力用のパラメータ
        self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
        #マウス入力の設定
        cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)
    
    #コールバック関数
    def __CallBackFunc(self, eventType, x, y, flags, userdata):
        
        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType    
        self.mouseEvent["flags"] = flags    

    #マウス入力用のパラメータを返すための関数
    def getData(self):
        return self.mouseEvent
    
    #マウスイベントを返す関数
    def getEvent(self):
        return self.mouseEvent["event"]                

    #マウスフラグを返す関数
    def getFlags(self):
        return self.mouseEvent["flags"]                

    #xの座標を返す関数
    def getX(self):
        return self.mouseEvent["x"]  

    #yの座標を返す関数
    def getY(self):
        return self.mouseEvent["y"]  

    #xとyの座標を返す関数
    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])
        
def give_coor(f):

    #表示するWindow名
    window_name = "input window"
    
    #画像の表示
    cv2.imshow(window_name, f)
    
    #コールバックの設定
    mouseData = mouseParam(window_name)
    
    while 1:
        cv2.waitKey(20)
        #左クリックがあったら表示
        if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
            m_point = mouseData.getPos()
            print(mouseData.getPos())
        #右クリックがあったら終了
        elif mouseData.getEvent() == cv2.EVENT_RBUTTONDOWN:
            break
            
    cv2.destroyAllWindows()            
    return m_point

def give_coorList(f):

    clickList = []

    #表示するWindow名
    window_name = "input window"
    
    #画像の表示
    cv2.namedWindow("input window", cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, f)
    
    #コールバックの設定
    mouseData = mouseParam(window_name)
    
    while 1:
        cv2.waitKey(20)
        # 左クリックがあったら座標をリストに入れる
        if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
            m_point = mouseData.getPos()
            # 座標がリストになければ追加する
            if clickList.count(m_point)==0:
                clickList.append(m_point)
            print(mouseData.getPos())
            ret = 0
            break
        #右クリックがあったら終了
        elif mouseData.getEvent() == cv2.EVENT_RBUTTONDOWN:
            ret = 1
            break
            
    cv2.destroyAllWindows()
    # 座標のリストを返す        
    return ret, clickList

if __name__ == "__main__":
    from tkinter import filedialog
    from pythonFile import click_pct, timestump, k_means
    import glob
    import os
    # ファイルダイアログからファイル選択
    typ = [('','*')] 
    dir = 'C:\\pg'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 
    #入力画像
    read = cv2.imread(path)
    give_coor(read)