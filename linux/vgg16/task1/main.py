import os, cv2, random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import ticker
from PIL import Image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical

TRAIN_DIR = "/media/koshiba/Data/sportConpetitive/vgg/train/"
TEST_DIR = "/media/koshiba/Data/sportConpetitive/vgg/test/"
OUTPUT_DIR = "/media/koshiba/Data/sportConpetitive/vgg/output/"

ROWS = 150
COLS = 150
CHANNELS = 3
print(os.listdir(TRAIN_DIR+'/*'))

train_refree = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'refree' in i]
train_player = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'player' in i]
train_ow = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'ow' in i]

test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)]
train_images = train_refree + train_player + train_ow

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
        image = read_image(image_file)
        
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
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5,5), activation='relu', kernel_initializer='he_normal', input_shape=(ROWS, COLS, CHANNELS)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, kernel_size=(5,5), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(60, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(3, activation='relu', kernel_initializer='he_normal'))
    model.summary()
    
    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    
    return model

model = judo_model()

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

print(nb_epochs)