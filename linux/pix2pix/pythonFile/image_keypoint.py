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

    # Draw the joints and bones
    for i, (pt, pid) in enumerate(zip(pts, person_ids)):
        cnt = 0
        #print(pid)
        frame1 = draw_points_and_skeleton(image, pt, joints_dict()['coco']['skeleton'], person_index=pid, points_color_palette='gist_rainbow', skeleton_color_palette='jet',points_palette_samples=10)
        
        for i,data in enumerate(pt):
            h = int(data[0])
            w = int(data[1]-10)
            #print(w, h)
            cv2.putText(frame1, str(i+1), (w, h), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if mask[h][w+10]==255:
                cnt += 1
            if cnt == 10:
                frame2 = draw_points_and_skeleton(image, pt, joints_dict()['coco']['skeleton'], person_index=pid, points_color_palette='gist_rainbow', skeleton_color_palette='jet',points_palette_samples=10)


    # Ouput the results
    # cv2.imwrite(args.output, frame)
    return pts, frame1, frame2

def draw(image, pt):
    # Model parameters
    nchannels=48
    njoints=17
    #person_ids = np.arange(len(pts), dtype=np.int32)
    
    frame = draw_points_and_skeleton(image, pt, joints_dict()['coco']['skeleton'], person_index=0, points_color_palette='gist_rainbow', skeleton_color_palette='jet',points_palette_samples=10)
    for i,data in enumerate(pt):
        h = int(data[0])
        w = int(data[1]-10)
        print(w, h)
        cv2.putText(frame, str(i+1), (w, h), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return frame

if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--input", "-i", help="target image", type=str, default="input.jpg")
    parse.add_argument("--output", "-o", help="output file name", type=str, default="output.jpg")
    args = parse.parse_args()