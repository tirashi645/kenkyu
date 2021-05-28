import cv2
import numpy as np
import random
import sys
from tkinter import filedialog

def todo(input_image_path):
    # 画像をグレースケールで読み込み
    gray_src = cv2.imread(input_image_path, 0)

    # 前処理（平準化フィルターを適用した場合）
    # 前処理が不要な場合は下記行をコメントアウト
    blur_src = cv2.GaussianBlur(gray_src, (5, 5), 2)

    # 二値変換
    # 前処理を使用しなかった場合は、blur_srcではなくgray_srcに書き換えるする
    mono_src = cv2.threshold(blur_src, 48, 255, cv2.THRESH_BINARY)[1]

    # ラベリング結果書き出し用に二値画像をカラー変換
    color_src01 = cv2.cvtColor(mono_src, cv2.COLOR_GRAY2BGR)
    color_src02 = cv2.cvtColor(mono_src, cv2.COLOR_GRAY2BGR)

    # ラベリング処理
    label = cv2.connectedComponentsWithStats(mono_src)
    print(label[2].shape)

    # オブジェクト情報を項目別に抽出
    n = label[0] - 1
    img_mask = label[1]
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)

    print(label[2][1:])
    maxIndex = np.argmax(label[2][1:], axis=0)[-1] + 1
    print(maxIndex)

    img_mask = np.where(img_mask == maxIndex, 255, 0).astype('uint8')

    # オブジェクト情報を利用してラベリング結果を画面に表示
    for i in range(n):
    
        # 各オブジェクトの外接矩形を赤枠で表示
        x0 = data[i][0]
        y0 = data[i][1]
        x1 = data[i][0] + data[i][2]
        y1 = data[i][1] + data[i][3]
        cv2.rectangle(color_src01, (x0, y0), (x1, y1), (0, 0, 255))
        cv2.rectangle(color_src02, (x0, y0), (x1, y1), (0, 0, 255))

        # 各オブジェクトのラベル番号と面積に黄文字で表示
        cv2.putText(color_src01, "ID: " +str(i + 1), (x1 - 20, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        cv2.putText(color_src01, "S: " +str(data[i][4]), (x1 - 20, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

        # 各オブジェクトの重心座標をに黄文字で表示
        cv2.putText(color_src02, "X: " + str(int(center[i][0])), (x1 - 30, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        cv2.putText(color_src02, "Y: " + str(int(center[i][1])), (x1 - 30, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

    # 結果の表示
    cv2.imshow("color_src01", color_src01)
    cv2.imshow("color_src02", color_src02)
    cv2.imshow("img_mask", img_mask)
    cv2.imwrite('E:/data/pix2pix/output/labeling.jpg', color_src01)
    cv2.imwrite('E:/data/pix2pix/output/labeling_mask.jpg', img_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remove_noise(gray_src):
    # 画像をグレースケールで読み込み
    # gray_src = cv2.imread(input_image_path, 0)

    # 前処理（平準化フィルターを適用した場合）
    # 前処理が不要な場合は下記行をコメントアウト
    blur_src = cv2.GaussianBlur(gray_src, (5, 5), 2)

    # 二値変換
    # 前処理を使用しなかった場合は、blur_srcではなくgray_srcに書き換えるする
    mono_src = cv2.threshold(blur_src, 48, 255, cv2.THRESH_BINARY)[1]

    # ラベリング処理
    label = cv2.connectedComponentsWithStats(mono_src)

    # オブジェクト情報を項目別に抽出
    img_mask = label[1]

    maxIndex = np.argmax(label[2][1:], axis=0)[-1] + 1

    img_mask = np.where(img_mask == maxIndex, 255, 0).astype('uint8')

    return img_mask

if __name__ == '__main__':

    # ファイルダイアログからファイル選択
    typ = [('','*')] 
    dir = 'C:\\pg'
    input_image_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
    todo(input_image_path)

