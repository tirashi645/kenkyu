import os
import sys
import argparse
import ast
import cv2
import time
import glob
import torch
from vidgear.gears import CamGear
import numpy as np

sys.path.insert(1, '/media/koshiba/Data/simple-HRNet')
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from misc.utils import find_person_id_associations

def main(camera_id=0, filename=None, hrnet_m='HRNet', hrnet_c=48, hrnet_j=17, hrnet_weights="/media/koshiba/Data/simple-HRNet/weights/pose_hrnet_w48_384x288.pth",
         hrnet_joints_set="coco", image_resolution='(384, 288)',
         single_person="store_true", use_tiny_yolo="store_true", disable_tracking="store_true", max_batch_size=16, disable_vidgear="store_true", save_video="store_true",
          video_format='MJPG', video_framerate=30, device=None):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    inputPath = '/media/koshiba/Data/simple-HRNet/inputData'
    hrnetPath = '/media/koshiba/Data/simple-HRNet'

    # print(device)

    image_resolution = ast.literal_eval(image_resolution)
    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'
    video_writer = None

    filePath = glob.glob(inputPath + '/*')

    for filename in filePath:
        videoName = filename.split('/')[-1][:-4]
        print(videoName)

        rotation_code = check_video_rotation(filename)
        video = cv2.VideoCapture(filename)
        assert video.isOpened()

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) #幅
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) #高
        frame_rate = int(video.get(cv2.CAP_PROP_FPS)) #FPS
        print(width, height, frame_rate)
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v') #mp4出力

        video_writer = cv2.VideoWriter('/media/koshiba/Data/simple-HRNet/outputData/' + videoName + '_output.mp4', fourcc, frame_rate, (width, height))

        if use_tiny_yolo:
            yolo_model_def= hrnetPath + "/models/detectors/yolo/config/yolov3-tiny.cfg"
            yolo_class_path= hrnetPath + "/models/detectors/yolo/data/coco.names"
            yolo_weights_path= hrnetPath + "/models/detectors/yolo/weights/yolov3-tiny.weights"
        else:
            yolo_model_def= hrnetPath + "/models/detectors/yolo/config/yolov3.cfg"
            yolo_class_path= hrnetPath + "/models/detectors/yolo/data/coco.names"
            yolo_weights_path= hrnetPath + "/models/detectors/yolo/weights/yolov3.weights"

        model = SimpleHRNet(
            hrnet_c,
            hrnet_j,
            hrnet_weights,
            model_name=hrnet_m,
            resolution=image_resolution,
            multiperson=not single_person,
            return_bounding_boxes=not disable_tracking,
            max_batch_size=max_batch_size,
            yolo_model_def=yolo_model_def,
            yolo_class_path=yolo_class_path,
            yolo_weights_path=yolo_weights_path,
            device=device
        )

        if not disable_tracking:
            prev_boxes = None
            prev_pts = None
            prev_person_ids = None
            next_person_id = 0

        while True:
            t = time.time()

            if filename is not None or disable_vidgear:
                ret, frame = video.read()
                if not ret:
                    break
                if rotation_code is not None:
                    frame = cv2.rotate(frame, rotation_code)
            else:
                frame = video.read()
                if frame is None:
                    break

            pts = model.predict(frame)

            if not disable_tracking:
                boxes, pts = pts

            if not disable_tracking:
                if len(pts) > 0:
                    if prev_pts is None and prev_person_ids is None:
                        person_ids = np.arange(next_person_id, len(pts) + next_person_id, dtype=np.int32)
                        next_person_id = len(pts) + 1
                    else:
                        boxes, pts, person_ids = find_person_id_associations(
                            boxes=boxes, pts=pts, prev_boxes=prev_boxes, prev_pts=prev_pts, prev_person_ids=prev_person_ids,
                            next_person_id=next_person_id, pose_alpha=0.2, similarity_threshold=0.4, smoothing_alpha=0.1,
                        )
                        next_person_id = max(next_person_id, np.max(person_ids) + 1)
                else:
                    person_ids = np.array((), dtype=np.int32)

                prev_boxes = boxes.copy()
                prev_pts = pts.copy()
                prev_person_ids = person_ids

            else:
                person_ids = np.arange(len(pts), dtype=np.int32)

            for i, (pt, pid) in enumerate(zip(pts, person_ids)):
                frame = draw_points_and_skeleton(frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=pid,
                                                points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                                points_palette_samples=10)

            fps = 1. / (time.time() - t)
            print('\rframerate: %f fps' % fps, end='')
            
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*video_format)  # video format
                video_writer = cv2.VideoWriter('/media/koshiba/Data/simple-HRNet/outputData/' + videoName + '_output.avi', fourcc, video_framerate, (frame.shape[1], frame.shape[0]))
            video_writer.write(frame)

        video_writer.release()
        print(person_ids)

def image_keypoint(frame, camera_id=0, filename=None, hrnet_m='HRNet', hrnet_c=48, hrnet_j=17, hrnet_weights="/media/koshiba/Data/simple-HRNet/weights/pose_hrnet_w48_384x288.pth",
         hrnet_joints_set="coco", image_resolution='(384, 288)',
         single_person="store_true", use_tiny_yolo="store_true", disable_tracking="store_true", max_batch_size=16, disable_vidgear="store_true", save_video="store_true",
          video_format='MJPG', video_framerate=30, device=None):

    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
        
    inputPath = '/media/koshiba/Data/simple-HRNet/inputData'
    hrnetPath = '/media/koshiba/Data/simple-HRNet'

    # print(device)

    image_resolution = ast.literal_eval(image_resolution)
    video_writer = None
    '''
    videoName = filename.split('/')[-1][:-4]
    print(videoName)

    rotation_code = check_video_rotation(filename)
    video = cv2.VideoCapture(filename)
    assert video.isOpened()

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) #幅
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) #高
    frame_rate = int(video.get(cv2.CAP_PROP_FPS)) #FPS
    print(width, height, frame_rate)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v') #mp4出力

    video_writer = cv2.VideoWriter('/media/koshiba/Data/simple-HRNet/outputData/' + videoName + '_output.mp4', fourcc, frame_rate, (width, height))
    '''

    if use_tiny_yolo:
        yolo_model_def= hrnetPath + "/models/detectors/yolo/config/yolov3-tiny.cfg"
        yolo_class_path= hrnetPath + "/models/detectors/yolo/data/coco.names"
        yolo_weights_path= hrnetPath + "/models/detectors/yolo/weights/yolov3-tiny.weights"
    else:
        yolo_model_def= hrnetPath + "/models/detectors/yolo/config/yolov3.cfg"
        yolo_class_path= hrnetPath + "/models/detectors/yolo/data/coco.names"
        yolo_weights_path= hrnetPath + "/models/detectors/yolo/weights/yolov3.weights"

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        model_name=hrnet_m,
        resolution=image_resolution,
        multiperson=not single_person,
        return_bounding_boxes=not disable_tracking,
        max_batch_size=max_batch_size,
        yolo_model_def=yolo_model_def,
        yolo_class_path=yolo_class_path,
        yolo_weights_path=yolo_weights_path,
        device=device
    )

    if not disable_tracking:
        prev_boxes = None
        prev_pts = None
        prev_person_ids = None
        next_person_id = 0
    '''
    while True:
        t = time.time()

        if filename is not None or disable_vidgear:
            ret, frame = video.read()
            if not ret:
                break
            if rotation_code is not None:
                frame = cv2.rotate(frame, rotation_code)
        else:
            frame = video.read()
            if frame is None:
                break
    '''
    pts = model.predict(frame)

    if not disable_tracking:
        boxes, pts = pts

    if not disable_tracking:
        if len(pts) > 0:
            if prev_pts is None and prev_person_ids is None:
                person_ids = np.arange(next_person_id, len(pts) + next_person_id, dtype=np.int32)
                next_person_id = len(pts) + 1
            else:
                boxes, pts, person_ids = find_person_id_associations(
                    boxes=boxes, pts=pts, prev_boxes=prev_boxes, prev_pts=prev_pts, prev_person_ids=prev_person_ids,
                    next_person_id=next_person_id, pose_alpha=0.2, similarity_threshold=0.4, smoothing_alpha=0.1,
                )
                next_person_id = max(next_person_id, np.max(person_ids) + 1)
        else:
            person_ids = np.array((), dtype=np.int32)

        prev_boxes = boxes.copy()
        prev_pts = pts.copy()
        prev_person_ids = person_ids

    else:
        person_ids = np.arange(len(pts), dtype=np.int32)

    for i, (pt, pid) in enumerate(zip(pts, person_ids)):
        frame = draw_points_and_skeleton(frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=pid,
                                        points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                        points_palette_samples=10)

    
    '''
    if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*video_format)  # video format
        video_writer = cv2.VideoWriter('/media/koshiba/Data/simple-HRNet/outputData/' + videoName + '_output.avi', fourcc, video_framerate, (frame.shape[1], frame.shape[0]))
    video_writer.write(frame)

    video_writer.release()
    print(person_ids)
    '''
    #print(pts)
    return pts, frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", "-d", help="open the camera with the specified id", type=int, default=0)
    parser.add_argument("--filename", "-f", help="open the specified video (overrides the --camera_id option)",
                        type=str, default=None)
    parser.add_argument("--hrnet_m", "-m", help="network model - 'HRNet' or 'PoseResNet'", type=str, default='HRNet')
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels (if model is HRNet), "
                                                "resnet size (if model is PoseResNet)", type=int, default=48)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="/media/koshiba/Data/simple-HRNet/weights/pose_hrnet_w48_384x288.pth")
    parser.add_argument("--hrnet_joints_set",
                        help="use the specified set of joints ('coco' and 'mpii' are currently supported)",
                        type=str, default="coco")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')
    parser.add_argument("--single_person",
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                             "multiperson detection)",
                        action="store_true")
    parser.add_argument("--use_tiny_yolo",
                        help="Use YOLOv3-tiny in place of YOLOv3 (faster person detection). Ignored if --single_person",
                        action="store_true")
    parser.add_argument("--disable_tracking",
                        help="disable the skeleton tracking and temporal smoothing functionality",
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--disable_vidgear",
                        help="disable vidgear (which is used for slightly better realtime performance)",
                        action="store_true")  # see https://pypi.org/project/vidgear/
    parser.add_argument("--save_video", help="save output frames into a video.", action="store_true")
    parser.add_argument("--video_format", help="fourcc video format. Common formats: `MJPG`, `XVID`, `X264`."
                                                     "See http://www.fourcc.org/codecs.php", type=str, default='MJPG')
    parser.add_argument("--video_framerate", help="video framerate", type=float, default=30)
    parser.add_argument("--device", help="device to be used (default: cuda, if available)."
                                         "Set to `cuda` to use all available GPUs (default); "
                                         "set to `cuda:IDS` to use one or more specific GPUs "
                                         "(e.g. `cuda:0` `cuda:1,2`); "
                                         "set to `cpu` to run on cpu.", type=str, default=None)
    args = parser.parse_args()
    main(**args.__dict__)
