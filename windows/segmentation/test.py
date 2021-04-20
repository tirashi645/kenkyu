from detectron2.data.detection_utils import read_image
from detectron2.model_zoo.model_zoo import ModelZooUrls
from detectron2.config import get_cfg
from demo.predictor import VisualizationDemo

img_path = 'input.jpg'
img = read_image(img_path, format="BGR")

for i, config_path in enumerate(ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX.keys()):
    # rpnとfast_rcnnは可視化対応していないので飛ばす
    if 'rpn' in config_path or 'fast_rcnn' in config_path:
        continue
    # config設定
    cfg = get_cfg()
    cfg.merge_from_file(f'configs/{config_path}')
    cfg.MODEL.WEIGHTS = ModelZooUrls.get(config_path)
    score_thresh = 0.5
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = score_thresh
    cfg.freeze()
    # 検出＆可視化
    demo = VisualizationDemo(cfg)
    predictions, visualized_output = demo.run_on_image(img)
    # ファイル出力
    dataset_name, algorithm = config_path.split("/")
    algorithm = algorithm.split('.')[0]
    visualized_output.save(f'out/{i:02d}-{dataset_name}-{algorithm}.jpg')