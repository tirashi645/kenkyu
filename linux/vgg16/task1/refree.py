import os, cv2, random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import ticker
from PIL import Image

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import models
from tensorflow.keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dense, Activation, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

#TRAIN_DIR = "/media/koshiba/Data/sportConpetitive/judo_data/refree2/"
TEST_DIR = "/media/koshiba/Data/sportConpetitive/judo_data/refree2/test/"
OUTPUT_DIR = "/media/koshiba/Data/sportConpetitive/refree/output/"

TRAIN_DIR = "/media/koshiba/Data/sportConpetitive/judo_data/refree_skeleton/"

ROWS = 150
COLS = 150
CHANNELS = 3

#print(os.listdir(TRAIN_DIR+'refree/'))

train_refree = [TRAIN_DIR+'ippon/' + i for i in os.listdir(TRAIN_DIR+'ippon/')]
train_player = [TRAIN_DIR+'wazaari/' + i for i in os.listdir(TRAIN_DIR+'wazaari/')]
train_ow = [TRAIN_DIR+'normal/' + i for i in os.listdir(TRAIN_DIR+'normal/')]
train_tmp = [TRAIN_DIR+'tmp/' + i for i in os.listdir(TRAIN_DIR+'tmp/')]

#test_refree = [TEST_DIR+'ippon/' + i for i in os.listdir(TEST_DIR+'ippon/')]
#test_player = [TEST_DIR+'wazaari/' + i for i in os.listdir(TEST_DIR+'wazaari/')]
#test_ow = [TEST_DIR+'normal/' + i for i in os.listdir(TEST_DIR+'normal/')]


#test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)]
train_images = train_refree + train_player + train_ow + train_tmp
#test_images = test_refree + test_player + test_ow

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

def tp(y_true, y_pred):
    return K.sum(K.round(y_true * y_pred)) * batch_size

#print(train_images)
train_data = prep_data(train_images)
#test_data = prep_data(test_images)

# 正規化
#train_data = train_data.astype('float32')
#train_data = train_data/255.0

# ラベルデータの作成
train_labels = []
for i in train_images:
    if 'ippon' in i:
        train_labels.append(1)
    elif 'wazaari' in i:
        train_labels.append(0)
    elif 'normal' in i:
        train_labels.append(0)
    elif 'tmp' in i:
        train_labels.append(0)

'''  
test_labels = []
for i in test_images:
    if 'ippon' in i:
        test_labels.append(0)
    elif 'wazaari' in i:
        test_labels.append(1)
    elif 'normal' in i:
        test_labels.append(1)
'''
  
train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=1)      
train_data, validation_data, train_labels, validation_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=1)

# convert to one-hot-label
train_labels = to_categorical(train_labels, 2)
test_labels = to_categorical(test_labels, 2)
#validation_labels = to_categorical(validation_labels, 2)

#学習用のImageDataGeneratorクラスの作成
augmentation_train_datagen = ImageDataGenerator(
    #左右反転
    horizontal_flip = True,
    #左右平行移動
    width_shift_range = 0.2,
    #ランダムにズーム
    zoom_range = 0.2,
    #スケーリング
    rescale = 1./255
    )
#学習用のバッチの生成
augmentation_train_data = augmentation_train_datagen.flow(train_data, train_labels, batch_size=32, seed=1234)
augmentation_validation_data = augmentation_train_datagen.flow(validation_data, validation_labels, batch_size=32, seed=1234)

# 最適化アルゴリズム
optimizer = 'SGD'
# 目的関数
objective = 'categorical_crossentropy'

# モデル構築
def judo_model():
    input_tensor = Input(shape=(ROWS, COLS, CHANNELS))
    #vgg16 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    top_model = models.Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    #top_model.add(vgg16)
    top_model.add(Flatten())
    top_model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
    #top_model.add(Dense(60, activation='relu', kernel_initializer='he_normal'))
    top_model.add(Dense(2, activation='softmax'))
    
    model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))
    
    for layer in top_model.layers[:15]:
        layer.trainable = False
    #vgg16.trainable = False
    '''
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5,5), activation='relu', kernel_initializer='he_normal', input_shape=(ROWS, COLS, CHANNELS)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, kernel_size=(5,5), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(60, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(2, activation='sigmoid', kernel_initializer='he_normal'))
    
    '''
    
    model.summary()
    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model

model = judo_model()

# number of epochs
epochs = 100
# batch_size
batch_size = 30

# monitor the trend of losses
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
# EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='c')

def run_judo_discriminator():
    history = LossHistory()
    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.3, verbose=1, shuffle=True, callbacks=[history, early_stopping])
    #model.fit_generator(augmentation_train_data, steps_per_epoch=int(len(train_data)/batch_size) , epochs=100, validation_data=augmentation_validation_data, validation_steps=int(len(validation_data)/batch_size), callbacks=[history, early_stopping])
    
    predictions = model.predict(test_data, verbose=1)
    return predictions, history

predictions, history = run_judo_discriminator()

'''
loss = history.losses
val_loss = history.val_losses

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Judo-Net trend')
plt.plot(loss, 'blue', label='training Loss')
plt.plot(val_loss, 'red', label='Validation Loss')
plt.xticks(range(0,epochs)[0::5])
plt.legend()
plt.savefig(OUTPUT_DIR + '/fig.jpg')
'''

score = model.evaluate(test_data, test_labels, verbose=1)
print('Test loss:', score[0])
print('Test acuuracy:', score[1])

model.save(OUTPUT_DIR + 'refree_model1.h5')
model.save_weights(OUTPUT_DIR + 'refree_model1_weight.h5')

#x_test = prep_data(test_images)
print(train_labels)
print(model.predict(test_data))
predict_prob = model.predict(test_data)
#print(model.predict_classes(x_test))
predict_classes=np.argmax(predict_prob,axis=1)
print(predict_classes)