import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

batch_size = 1024
epochs = 1024 * 6
max_number = 100

def calculate(arr):
    return arr[0] + arr[1] 

def generate_data(size,elementSize=2):
    return np.random.randint(1, max_number + 1, size = (size,elementSize)) / (max_number * 1.0)

def generate_x(input):
    return input

def generate_y(input):
    ret = np.array([([0.0] * (max_number * 2))] * len(input)) 

    for i in range(len(input)):
        n = int(calculate(input[i]) * (max_number * 1.0)) - 1
        ret[i][n] = 1.0

    return ret

def generate_dnn_model():

    # relu
    # sigmoid
    # softmax
    # tanh

    sigma_function = 'tanh' 
    k_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05,seed=None)
    model = keras.Sequential([keras.Input(shape=(2)),
            layers.Dense(max_number / 2, activation=sigma_function, kernel_initializer=k_initializer), 
            layers.Dense(max_number , activation="relu", kernel_initializer=k_initializer), 
            layers.Dense(max_number * 2, activation = "softmax")])
    
    model.summary()
    l = keras.losses.CategoricalCrossentropy()
    op = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss=l, optimizer = op,  metrics=["accuracy"])
    return model

def train_model(model): 
    for x in range(epochs):
        data = generate_data(batch_size) 
        x_train = generate_x(data)
        y_train = generate_y(x_train)
        loss = model.train_on_batch(x_train, y_train)
        print("training:",int((x + 0.0) / epochs * 100), "loss:", round(loss[0],4), "accuracy:", round(loss[1],4), end='\r')

    print("")
    print("finish training")
    print("=================================")

model = generate_dnn_model()
train_model(model)

# test
while True:
    s1 = input()
    s2 = input()
    x1 = float(s1) / (max_number * 1.0)
    x2 = float(s2) / (max_number * 1.0)
    predict_value = model.predict(np.array([[x1,x2]]))
    idx = np.argmax(predict_value[0]) + 1
    print(s1,"+",s2,"=",idx)