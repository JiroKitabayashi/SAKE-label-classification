# coding:utf-8
```
モデルを生成するためのコード
modelsから特定の画像を
```

import keras
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import glob

folder = ["dassai_images", "kubota_images"]
image_size = 50

X = []
Y = []

for index, name in enumerate(folder):
    dir = "../images/" + name
    files = glob.glob(dir + "/*.jpg")
    for file in files:
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)   # 正解ラベル (1か2)

X = np.array(X) # nparray形式に変換
Y = np.array(Y) 

X = X.astype('float32')  # 不動点小数型に変換
X = X / 255.0
Y = np_utils.to_categorical(Y, 2) # 正解ラベルの形式を変換します。例：獺祭[1,0]、久保田[0,1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20) # トレーニングのサイズを全体の2割に設定


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=200)

print(model.evaluate(X_test, y_test))

json_string = model.to_json() # モデルの保存
open('../models/ponsh_model01.json', 'w').write(json_string)
model.save_weights('../weights/ponsh_weights01.h5') # 重みデータの保存

x = []
x_test = Image.open("../dassai1.jpg")
x_test = x_test.convert("RGB")
x_test = x_test.resize((image_size, image_size))
x_data = np.asarray(x_test)
x.append(x_data)
x = np.array(x)
x = x.astype('float32')
x = x / 255.0
print("dassai1")
print(model.predict(x))

x = []
x_test = Image.open("../dassai2.jpg")
x_test = x_test.convert("RGB")
x_test = x_test.resize((image_size, image_size))
x_data = np.asarray(x_test)
x.append(x_data)
x = np.array(x)
x = x.astype('float32')
x = x / 255.0
print("dassai2")
print(model.predict(x))

x = []
x_test = Image.open("../dassai3.jpg")
x_test = x_test.convert("RGB")
x_test = x_test.resize((image_size, image_size))
x_data = np.asarray(x_test)
x.append(x_data)
x = np.array(x)
x = x.astype('float32')
x = x / 255.0
print("dassai3")
print(model.predict(x))

x = []
x_test = Image.open("../kubota.jpg")
x_test = x_test.convert("RGB")
x_test = x_test.resize((image_size, image_size))
x_data = np.asarray(x_test)
x.append(x_data)
x = np.array(x)
x = x.astype('float32')
x = x / 255.0
print("kubota")
print(model.predict(x))

