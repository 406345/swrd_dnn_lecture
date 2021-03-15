import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import struct
import cv2
import msnit

batch_size = 100
epochs = 10 

def build_model(x, y):
    # relu
    # sigmoid
    # softmax
    # tanh
    sigma_function = 'sigmoid'

    model = keras.Sequential([
        keras.Input(shape=(28 * 28)),
        layers.Dense(16, bias_initializer='one', activation=sigma_function),
        layers.LayerNormalization(),
        layers.Dense(16, bias_initializer='one', activation=sigma_function),
        layers.LayerNormalization(),
        layers.Dense(10, bias_initializer='one', activation="softmax"),
    ])

    model.summary()
    adam = tf.keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model, x, y 

def train_model(x, y, predict = False):
    model, x, y = build_model(x, y)

    if predict :
        model.load_weights('./dnn_model.h5')
        return model

    model.fit(
        x=x,
        y=y,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
    )
    print("finish training")
    print("=================================")
    model.save('./dnn_model.h5')

    return model


labels, imgs = msnit.msnit_load()
model = train_model(imgs, labels, False)

# test
while True:
    print("Input your image: ")
    k = input()
    img_file = R'./handwriting.bmp'
    img = cv2.imread(img_file)
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    input_x = (np.array(img) / 255.).reshape(1, 28 * 28).tolist()[0]
    predict_value = model.predict([input_x])

    print('predict:', np.argmax(predict_value[0]))
    for n in range(10):
        print('number:', n, ' possibility:', predict_value[0][n])

    print('==================================')
