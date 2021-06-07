import cv2
import argparse
import os
import sys
import cv2
import numpy as np

sys.path.insert(1, '/media/koshiba/Data/simple-HRNet')
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from misc.utils import find_person_id_associations

def get_keypoint(image, mask):
    # Model parameters
    nchannels=48
    njoints=17

    # Creat the HRNet model
    model = SimpleHRNet(nchannels, njoints, "/media/koshiba/Data/simple-HRNet/weights/pose_hrnet_w48_384x288.pth")

    # Load the input image
    #image = cv2.imread(args.input, cv2.IMREAD_COLOR)

    # Perform the prediction for pose estimation
    pts = model.predict(image)
    person_ids = np.arange(len(pts), dtype=np.int32)
    frame1 = image

    # Draw the joints and bones
    print(pts)
    for i, (pt, pid) in enumerate(zip(pts, person_ids)):
        cnt = 0
        #print(pid)
        #frame1 = draw_points_and_skeleton(image, pt, joints_dict()['coco']['skeleton'], person_index=pid, points_color_palette='gist_rainbow', skeleton_color_palette='jet',points_palette_samples=10)
        frame1 = draw(frame1, pt)


    # Ouput the results
    # cv2.imwrite(args.output, frame)
    return pts, frame1

def draw(image, pt):
    # Model parameters
    nchannels=48
    njoints=17
    line = [[1,2],[1,3],[1,6],[1,7],[6,7],[3,5],[2,4],[6,8],[6,12],[7,9],[7,13],[8,10],[9,11],[12,13],[12,14],[13,15],[14,16],[15,17]]
    cp = ['p' if i[2]>0.5 else 'f' for i in pt]
    #person_ids = np.arange(len(pts), dtype=np.int32)
    
    #frame = draw_points_and_skeleton(image, pt, joints_dict()['coco']['skeleton'], person_index=0, points_color_palette='gist_rainbow', skeleton_color_palette='jet',points_palette_samples=10)
    for i,data in enumerate(pt):
        h = int(data[0])
        w = int(data[1]-10)
        #print(w, h)
        if cp[i]=='p':
            cv2.circle(image, (w, h), 5, (255, 255, 255), thickness=-1)
            cv2.putText(image, str(i+1), (w, h), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
    for point in line:
        s = point[0]-1
        t = point[1]-1
        if cp[s]=='p' and cp[t]=='p':
            cv2.line(image, (int(pt[s][1]), int(pt[s][0])), (int(pt[t][1]), int(pt[t][0])), (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    return image

if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--input", "-i", help="target image", type=str, default="input.jpg")
    parse.add_argument("--output", "-o", help="output file name", type=str, default="output.jpg")
    args = parse.parse_args()