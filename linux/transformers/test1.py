from transformers import DetrFeatureExtractor, DetrForObjectDetection, DetrForSegmentation
from PIL import Image
import requests
import glob
import matplotlib.pyplot as plt
import torch
#file_path = '/content/drive/My Drive/Colab Notebooks/CenterNet/images/' + fileName + '.mp4' #動画ファイルのパス
save_path = '/media/koshiba/Data/transformers/data/'

# imageフォルダ内のファイルパスをすべて取得する
image_list = glob.glob('/media/koshiba/Data/simple-HRNet/inputData/**/*.mp4')
print(image_list)

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        cl = p.argmax()
        if model.config.id2label[cl.item()]=='person':
            centerX = (xmin + xmax) / 2
            centerY = (ymin + ymax) / 2
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        fill=False, color=c, linewidth=3))
            ax.scatter(centerX, centerY, color=c)
            
            text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.show()

def plot_save(pil_img, prob, boxes, image_name):
    for i, p, (xmin, ymin, xmax, ymax)in enumerate(zip(prob, boxes.tolist())):
        cl = p.argmax()
        if model.config.id2label[cl.item()]=='person':
            if int(p[cl])>0.95:
                im = pil_img.crop((xmin, ymin, xmax, ymax))
                im.save(save_path + str(i) + '_' + image_name + '.jpg')

for image_path in image_list:
    image_name = image_path.split('/')[-1]
    image = Image.open(image_path)
    #url = "http://images.cocodataset.org/train2014/COCO_train2014_000000384029.jpg"
    #image = Image. open( requests. get( url, stream = True). raw)
    # 
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # 分類名やバウンディングボックスを描画する

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    
    # keep only predictions of queries with 0.9+ confidence (excluding no-object class)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    # rescale bounding boxes
    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]

    #plot_results(image, probas[keep], bboxes_scaled)
    plot_save(image, probas[keep], bboxes_scaled, image_name)