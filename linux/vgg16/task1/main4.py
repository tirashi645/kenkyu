import os, cv2, random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import ticker
from PIL import Image
from numpy.lib.function_base import average

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import models
from tensorflow.keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dense, Activation, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


TRAIN_DIR = "/media/koshiba/Data/sportConpetitive/refree/train/"
TEST_DIR = "/media/koshiba/Data/sportConpetitive/refree/test/"
OUTPUT_DIR = "/media/koshiba/Data/sportConpetitive/refree/output/"

ROWS = 50
COLS = 50
CHANNELS = 3

#print(os.listdir(TRAIN_DIR+'refree/'))

train_refree = [TRAIN_DIR+'ippon/' + i for i in os.listdir(TRAIN_DIR+'ippon/')]
train_player = [TRAIN_DIR+'wazaari/' + i for i in os.listdir(TRAIN_DIR+'wazaari/')]
train_ow = [TRAIN_DIR+'normal/' + i for i in os.listdir(TRAIN_DIR+'normal/')]

test_refree = [TEST_DIR+'ippon/' + i for i in os.listdir(TEST_DIR+'ippon/')]
test_player = [TEST_DIR+'wazaari/' + i for i in os.listdir(TEST_DIR+'wazaari/')]
test_ow = [TEST_DIR+'normal/' + i for i in os.listdir(TEST_DIR+'normal/')]


#test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)]
train_images = train_refree + train_player + train_ow
test_images = test_refree + test_player + test_ow[::5]

random.shuffle(train_images)

# 画像をリサイズして統一
def read_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return cv2.resize(image, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

# 各データの準備
def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)
    
    for i, image_file in enumerate(images):
        #print(image_file)
        image = read_image(image_file)
        
        data[i] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')/255.0
    
    return data


#print(train_images)
train_data = prep_data(train_images)
test_data = prep_data(test_images)

# 正規化
#train_data = train_data.astype('float32')
#train_data = train_data/255.0

# ラベルデータの作成
train_labels = []
for i in train_images:
    if 'ippon' in i:
        train_labels.append(0)
    elif 'wazaari' in i:
        train_labels.append(1)
    elif 'normal' in i:
        train_labels.append(2)
        
test_labels = []
for i in test_images:
    if 'ippon' in i:
        test_labels.append(0)
    elif 'wazaari' in i:
        test_labels.append(1)
    elif 'normal' in i:
        test_labels.append(2)

y_labels = test_labels

# convert to one-hot-label
train_labels = to_categorical(train_labels, 3)
#validation_labels = to_categorical(validation_labels, 3)
test_labels = to_categorical(test_labels, 3)

model = load_model(OUTPUT_DIR + 'judo_model4.h5')
model.load_weights(OUTPUT_DIR + 'judo_model4_weight.h5')

x_test = prep_data(test_images)
print(model.predict(x_test))
predict_prob = model.predict(x_test)
#print(model.predict_classes(x_test))
predict_classes=np.argmax(predict_prob,axis=1)
print(predict_classes)

for i, data in enumerate(test_images):
    image = cv2.imread(data)
    if predict_classes[i]==0:
        cv2.imwrite("/media/koshiba/Data/sportConpetitive/judo_data/output2/ippon/"+data.split('/')[-1], image)
    elif predict_classes[i]==1:
        cv2.imwrite("/media/koshiba/Data/sportConpetitive/judo_data/output2/waza/"+data.split('/')[-1], image)
    elif predict_classes[i]==2:
        cv2.imwrite("/media/koshiba/Data/sportConpetitive/judo_data/output2/ow/"+data.split('/')[-1], image)
        
score = model.evaluate(test_data, test_labels, verbose=1)
print('Test loss:', score[0])
print('Test acuuracy:', score[1])
#print('Accuracy:',accuracy_score(y_labels,predict_classes, average='weighted'))
print('Precision:', precision_score(y_labels,predict_classes, average='weighted'))
print('Recall:', recall_score(y_labels,predict_classes, average='weighted'))
print('F1 score:', f1_score(y_labels,predict_classes, average='weighted'))