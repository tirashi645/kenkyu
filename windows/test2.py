from os import name
import cv2
from matplotlib import pyplot as plt
from tkinter import filedialog

def match(from_img, to_img):
    akaze = cv2.AKAZE_create()
    shift = cv2.xfeatures2d.SIFT_create()
    orb = cv2.ORB_create()
    # 各画像の特徴点を取る
    from_key_points, from_descriptions = akaze.detectAndCompute(from_img, None)
    to_key_points, to_descriptions = akaze.detectAndCompute(to_img, None)
    
    # 2つの特徴点をマッチさせる
    bf_matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, True)
    matches = bf_matcher.match(from_descriptions, to_descriptions)
    
    # 特徴点を同士をつなぐ
    match_img = cv2.drawMatches(
        from_img, from_key_points, to_img, to_key_points, 
        matches,  None, flags=2
    )
    
    return match_img

if __name__ == '__main__':
    # ファイルダイアログからファイル選択
    typ = [('','*')] 
    dir = 'C:\\pg'
    image_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
    image_path2 = filedialog.askopenfilename(filetypes = typ, initialdir = dir)

    img_origin = cv2.imread(image_path, 1)
    img_moto = img_origin.copy()
    img_hikaku = cv2.imread(image_path2, 1)
    img_akaze = img_origin.copy()
    img_shift = img_origin.copy()
    img_orb = img_origin.copy()

    img = cv2.bitwise_not(img_origin)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, img_binary = cv2.threshold(img_gray, 220, 255,cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    img_contour = cv2.drawContours(img_origin, contours, -1, (0, 255, 0), 5)

    # グレースケール変換
    from_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 特徴点抽出 (AKAZE)
    akaze = cv2.AKAZE_create()
    shift = cv2.xfeatures2d.SIFT_create()
    orb = cv2.ORB_create()
    akaze_key_points, akaze_descriptions = akaze.detectAndCompute(from_img, None)
    shift_key_points, shift_descriptions = shift.detectAndCompute(from_img, None)
    orb_key_points, orb_descriptions = orb.detectAndCompute(from_img, None)

    # キーポイントの表示
    akaze_img = cv2.drawKeypoints(img_akaze, akaze_key_points, None, flags=4)
    shift_img = cv2.drawKeypoints(img_shift, akaze_key_points, None, flags=4)
    orb_img = cv2.drawKeypoints(img_orb, akaze_key_points, None, flags=4)
    #extraceted_img = cv2.drawKeypoints(img_akaze, akaze_key_points, None)

    matching_img = match(img_moto, img_hikaku)


    cv2.imwrite("E:/data/tmp/img_contour.jpg",img_contour)
    cv2.imwrite("E:/data/tmp/akaze2.jpg",akaze_img)
    cv2.imwrite("E:/data/tmp/shift2.jpg",shift_img)
    cv2.imwrite("E:/data/tmp/orb2.jpg",orb_img)
    cv2.imwrite("E:/data/tmp/akaze_MATCHING.jpg",matching_img)