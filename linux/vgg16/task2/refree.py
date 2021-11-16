import os, cv2, random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import ticker
from PIL import Image

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dense, Activation, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image

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
test_images = test_refree + test_player + test_ow

random.shuffle(train_images)

# 画像をリサイズして統一
def read_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    
    return cv2.resize(image, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

# 各データの準備
def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)
    
    for i, image_file in enumerate(images):
        print(image_file)
        image = read_image(image_file)
        image = cv2.resize(image, (ROWS, COLS))
        
        data[i] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return data



print(train_images)
train_data = prep_data(train_images)
test_data = prep_data(test_images)

# 正規化
train_data = train_data.astype('float32')
train_data = train_data/255.0

# ラベルデータの作成
train_labels = []
for i in train_images:
    if 'ippon' in i:
        train_labels.append(0)
for i in train_images:
    if 'wazaari' in i:
        train_labels.append(1)
for i in train_images:
    if 'normal' in i:
        train_labels.append(2)
        
test_labels = []
for i in test_images:
    if 'ippon' in i:
        test_labels.append(0)
for i in test_images:
    if 'wazaari' in i:
        test_labels.append(1)
for i in test_images:
    if 'normal' in i:
        test_labels.append(2)

# convert to one-hot-label
train_labels = to_categorical(train_labels, 3)
test_labels = to_categorical(test_labels, 3)

# 最適化アルゴリズム
optimizer = 'SGD'
# 目的関数
objective = 'categorical_crossentropy'

# モデル構築
def judo_model():
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5,5), activation='relu', kernel_initializer='he_normal', input_shape=(ROWS, COLS, CHANNELS)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, kernel_size=(5,5), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(60, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='relu', kernel_initializer='he_normal'))
    model.summary()
    
    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    
    return model


def res_model():
    input_tensor = Input(shape=(ROWS, COLS, CHANNELS))
    vgg16 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(3, activation='relu', kernel_initializer='he_normal'))
    
    model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))
    
    for layer in model.layers[:15]:
        layer.trainable = False
    
    model.summary()
    
    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    
    return model

model = res_model()

# number of epochs
epochs = 30
# batch_size
batch_size = 20

# monitor the trend of losses
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
# EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

def run_judo_discriminator():
    history = LossHistory()
    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1, shuffle=True, callbacks=[history])
    
    predictions = model.predict(test_data, verbose=1)
    return predictions, history

predictions, history = run_judo_discriminator()

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

score = model.evaluate(test_data, test_labels, verbose=1)
print('Test loss:', score[0])
print('Test acuuracy:', score[1])

model.save(OUTPUT_DIR + 'judo_model2.h5')
model.save_weights(OUTPUT_DIR + 'judo_model2_weight.h5')
#print(nb_epochs)
x_test = prep_data(test_images)
print(train_labels)
print(model.predict(x_test))
print(model.predict_classes(x_test))
