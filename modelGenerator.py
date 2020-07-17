import numpy as np
import os

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Conv2D, MaxPool2D

os.environ['TF_DISABLE_MKL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
import tensorflow.keras as keras
from datetime import datetime
import matplotlib.pyplot as plt

from fileImageLoader import  load_dataset


#========================MODEL PARAMETER=====================
target_resolution = (64, 64)
numberOfType = 4
epochs = 500
#============================================================

#========================MODELS NAME=========================
TAG_MLP = "mlpmodel.keras"
TAG_LINEAR = "linearmodel.keras"
TAG_DENSE_RES_NN = "denseresnnmodel.keras"
TAG_DENSE_U_NN  = "denseunnmodel.keras"
TAG_CNN  = "cnn.keras"
#============================================================


def create_linear_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(numberOfType, activation=sigmoid))
    model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['accuracy'])
    return model


def create_mlp_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation=tanh))
    model.add(Dense(256, activation=tanh))
    model.add(Dense(numberOfType, activation=sigmoid))
    model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['accuracy'])
    return model


def create_conv_nn_model():
    model = Sequential()

    model.add(Conv2D(4, (3, 3), padding='same', activation=relu,
                     input_shape=( target_resolution[0],target_resolution[1], 3)))
    model.add(MaxPool2D((2, 2)))

    model.add(Conv2D(8, (3, 3), padding='same', activation=relu))
    model.add(MaxPool2D((2, 2)))

    model.add(Conv2D(16, (3, 3), padding='same', activation=relu ))


    model.add(Flatten())
    model.add(Dense(64, activation=tanh))
    model.add(Dense(numberOfType, activation=sigmoid))
    model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['accuracy'])
    return model


def create_dense_res_nn_model():
    input_tensor = keras.layers.Input((target_resolution[0], target_resolution[1], 3))

    previous_tensor = Flatten()(input_tensor)

    next_tensor = Dense(64, activation=relu)(previous_tensor)

    previous_tensor = keras.layers.Concatenate()([previous_tensor, next_tensor])

    next_tensor = Dense(64, activation=relu)(previous_tensor)

    previous_tensor = keras.layers.Concatenate()([previous_tensor, next_tensor])

    next_tensor = Dense(64, activation=relu)(previous_tensor)

    previous_tensor = keras.layers.Concatenate()([previous_tensor, next_tensor])

    next_tensor = Dense(64, activation=tanh)(previous_tensor)
    next_tensor = Dense(numberOfType, activation=sigmoid)(next_tensor)

    model = keras.models.Model(input_tensor, next_tensor)

    model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['accuracy'])
    return model


def create_dense_u_nn_model():
    input_tensor = keras.layers.Input((target_resolution[0], target_resolution[1]))

    flattened_input = Flatten()(input_tensor)

    output1 = Dense(64, activation=relu)(flattened_input)

    output2 = Dense(64, activation=relu)(output1)

    output3 = Dense(64, activation=relu)(output2)

    previous_tensor = keras.layers.Concatenate()([output1, output3])

    output4 = Dense(64, activation=tanh)(previous_tensor)

    previous_tensor = keras.layers.Concatenate()([flattened_input, output4])

    next_tensor = Dense(numberOfType, activation=sigmoid)(previous_tensor)

    model = keras.models.Model(input_tensor, next_tensor)

    model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['accuracy'])
    return model

def createDataAugmentor(x_train, y_train) :
    datagen = ImageDataGenerator(rotation_range= 90,
                                 zoom_range=[0.5,1.0], width_shift_range=[-32,32])
    it = datagen.flow(x_train, y_train, batch_size=16)

    return it

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = load_dataset()


    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)


    it_train = createDataAugmentor(x_train, y_train)

    #model = create_linear_model()
    model = create_mlp_model()
    #model = create_conv_nn_model()
    #model = create_dense_res_nn_model()
    #model = create_dense_u_nn_model()

    model.build(x_train.shape)

    model.summary()
    logdir = "logs/scalars/" + datetime.now().strftime("%Y-%m-%d-%H%M%S")+"AUGMENTED_DENSEUNN_500"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    logs = model.fit(x_train, y_train, epochs=epochs, batch_size=16, validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback])

    print(logs.history.keys())


    tag_name = TAG_MLP

    model.save(f"./models/{tag_name}")
    print(f"Model Augmenté enregistré ! \n nom du model : {tag_name}")

    fig, (ax0, ax1) = plt.subplots(nrows=2, constrained_layout=True)

    # Graph Accuracy
    ax0.set_title("Accuracy")
    ax0.plot(logs.history['accuracy'])
    ax0.plot(logs.history['val_accuracy'])

    # Graph Loss
    ax1.set_title("Loss")
    ax1.plot(logs.history['loss'])
    ax1.plot(logs.history['val_loss'])

    plt.show()
