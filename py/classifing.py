# coding:utf-8
```
学習済みモデルを呼び出してきて、画像を判別する。
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

image_size = 50
# モデルを読み込む
model = model_from_json(open('../models/ponsh_model.json').read())

# 学習結果を読み込む
model.load_weights('../weights/ponsh_weights.h5')

model.summary();

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