import os, cv2, random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import ticker
from PIL import Image

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dense, Activation, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

TRAIN_DIR = "/media/koshiba/Data/sportConpetitive/train/"
TEST_DIR = "/media/koshiba/Data/sportConpetitive/test/"
OUTPUT_DIR = "/media/koshiba/Data/sportConpetitive/vgg16/output/"

ROWS = 50
COLS = 50
CHANNELS = 3
#print(os.listdir(TRAIN_DIR+'refree/'))

train_refree = [TRAIN_DIR+'refree/' + i for i in os.listdir(TRAIN_DIR+'refree/')]
train_player = [TRAIN_DIR+'player/' + i for i in os.listdir(TRAIN_DIR+'player/')]
train_ow = [TRAIN_DIR+'ow/' + i for i in os.listdir(TRAIN_DIR+'ow/')]

test_refree = [TEST_DIR+'refree/' + i for i in os.listdir(TEST_DIR+'refree/')]
test_player = [TEST_DIR+'player/' + i for i in os.listdir(TEST_DIR+'player/')]
test_ow = [TEST_DIR+'ow/' + i for i in os.listdir(TEST_DIR+'ow/')]

#test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)]
train_images = train_refree + train_player + train_ow[::10]
test_images = test_refree + test_ow# + test_player + test_ow

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
        
        data[i] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')/255.0
    
    return data

def tp(y_true, y_pred):
    return K.sum(K.round(y_true * y_pred)) * batch_size

#print(train_images)
train_data = prep_data(train_images)
test_data = prep_data(test_images)

# 正規化
#train_data = train_data.astype('float32')
#train_data = train_data/255.0

# ラベルデータの作成
train_labels = []
for i in train_images:
    if 'refree' in i:
        train_labels.append(0)
for i in train_images:
    if 'player' in i:
        train_labels.append(1)
for i in train_images:
    if 'ow' in i:
        train_labels.append(2)
        
test_labels = []
for i in test_images:
    if 'refree' in i:
        test_labels.append(0)
for i in test_images:
    if 'player' in i:
        test_labels.append(1)
for i in test_images:
    if 'ow' in i:
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
    input_tensor = Input(shape=(ROWS, COLS, CHANNELS))
    vgg16 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(120, activation='relu', kernel_initializer='he_normal'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(3, activation='relu', kernel_initializer='he_normal'))
    
    model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))
    
    for layer in model.layers[:15]:
        layer.trainable = False
    
    model.summary()
    
    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    
    return model

model = judo_model()

# number of epochs
epochs = 30
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

x_test = prep_data(test_images)
print(model.predict(x_test))
x_predict = []
for i in model.predict(model.predict(x_test)):
    for cnt, j in enumerate(i):
        if j==1:
            x_predict.append(cnt)
print(train_labels)