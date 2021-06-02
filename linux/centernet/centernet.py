import os
import sys
from os.path import join
import cv2
import glob
import time
centerNetPath = '/media/koshiba/Data/CenterNet'

sys.path.insert(0, join(centerNetPath, 'src/lib'))
sys.path.append(join(centerNetPath, 'src'))
sys.path.append(join(centerNetPath, 'src/lib/models/networks/DCNv2'))


from detectors.detector_factory import detector_factory
from opts import opts

MODEL_PATH = centerNetPath + '/models/multi_pose_dla_3x.pth' #モデルのパス
TASK = 'multi_pose' #骨格点検出
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)

#print('入力する動画のファイル名を入力してください(拡張子は不要です)')
#fileName = input()

#file_path = centerNetPath + '/images/' + fileName + '.mp4' #動画ファイルのパス
save_path = centerNetPath + '/output'

# imageフォルダ内のファイルパスをすべて取得する
image_list = glob.glob(centerNetPath + '/images/*')

for file_path in image_list:
  start = time.time()
  # video名を取得
  fileName = file_path.split('/')[-1]
  print()
  fileName = fileName[:fileName.rfind(".")]

  if not os.path.exists(save_path):
    os.makedirs(save_path)
  if os.path.exists(file_path):
    cap = cv2.VideoCapture(file_path) #動画の読み込み
    line = [[1,2],[1,3],[6,7],[3,5],[2,4],[6,8],[6,12],[7,9],[7,13],[8,10],[9,11],[12,13],[12,14],[13,15],[14,16],[15,17]] # 特徴点を繋げる線

    #出力用設定
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #幅
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #高
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) #FPS
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v') #mp4出力

    out = cv2.VideoWriter(save_path + '/' + fileName + '_output.mp4', fourcc, frame_rate, (width, height))

    while True:
      gg, img = cap.read() #フレームの取り出し
      if gg: #取り出せた場合
        rets = detector.run(img)['results'] #骨格点検出
        ret = rets.get(1) #人のキーを指定

        for rret in ret: #人数分繰り返し
          if rret[4] > 0.5: #閾値を0.5でセット
            #人を矩形で囲う
            cv2.rectangle(img, (int(rret[0]), int(rret[1])), (int(rret[2]), int(rret[3])), (0, 255, 0, 255), 2)

            #骨格点に点を描く
            num = int((len(rret)-5)/2+1)
            for i in range(num):
              cv2.putText(img, str(i), (int(rret[2*(i-1)+5]), int(rret[2*(i-1)+6]-5)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)
              cv2.circle(img, (int(rret[2*(i-1)+5]), int(rret[2*(i-1)+6])), 5, (255, 255, 255), thickness=-1)
            #特徴点を線でつなぐ
            for point in line:
              s = point[0]
              t = point[1]
              cv2.line(img, (int(rret[2*(s-1)+5]), int(rret[2*(s-1)+6])), (int(rret[2*(t-1)+5]), int(rret[2*(t-1)+6])), (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        out.write(img)
      else: #フレームが取り出せない（終点）
        break
    out.release() 
    # 終了時間の表示
    time_taken = time.time() - start
    print('video:{0}   time:{1}[sec]'.format(fileName, time_taken))
  else:
    print('ファイルが存在しません')