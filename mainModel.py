import numpy as np

import os

from tensorflow.keras.callbacks import TensorBoard

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mean_squared_error
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import keras
from keras.layers.core import Activation,Dropout, Flatten, Dense


from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Conv2D, MaxPool2D
target_resolution = (4, 4)



def color_treatment (image):
    """Analyse une image et trouve le centre de la forme si elle existe.
        Prend en parametre :
        - image : l'image a analyser

        Renvoie : Une image binaire, composé uniquement de noir et de blanc"""

    ####### Les prochaines lignes permettent de travailler l'image pour #######
    #######            la rendre plus facilement analysable             #######

    # On transforme l'image en niveaux de gris
    gris = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2GRAY)
    #gris = image
    # On floute l'image
    flou = cv2.GaussianBlur(gris, (5, 5), 1)

    # On transforme en image binaire
    ret, binaire = cv2.threshold(flou, 140, 255, cv2.THRESH_BINARY_INV)

    # On elimine les bruits restants
    masque = cv2.erode(binaire, None, iterations=2)
    masque = cv2.dilate(masque, None, iterations=2)

    ###########################################################################
    ###########################################################################
    return masque

def load_dataset():
    Ximgs = []
    y_train = []

    pathTrain = "./dataset/train/"
    pathTest = "./dataset/test/"
    target_resolution = (64, 64 )
    for file in os.listdir(pathTrain+"hotdog/"):

        #selectedImage = Image.open(pathTrain+f"hotdog/{file}").resize(target_resolution).convert('RGB')
        selectedImage = cv2.imread(pathTrain+f"hotdog/{file}")
        resized_image = cv2.resize(selectedImage,target_resolution)
        #resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        masque = color_treatment(resized_image) / 255

        Ximgs.append(
            np.array(masque))
        y_train.append([1, 0, 0, 0])
    for file in os.listdir(pathTrain+"pizza/"):

        #selectedImage = Image.open(pathTrain + f"pizza/{file}").resize(target_resolution).convert('RGB')
        selectedImage = cv2.imread(pathTrain + f"pizza/{file}")
        resized_image = cv2.resize(selectedImage, target_resolution)
        masque = color_treatment(resized_image) / 255

        Ximgs.append(
            np.array(masque))
        y_train.append([0, 1, 0, 0])
    for file in os.listdir(pathTrain+"burger/"):

        #selectedImage = Image.open(pathTrain + f"burger/{file}").resize(target_resolution).convert('RGB')
        selectedImage = cv2.imread(pathTrain + f"burger/{file}")
        resized_image = cv2.resize(selectedImage, target_resolution)
        masque = color_treatment(resized_image) / 255

        Ximgs.append(
            np.array(masque))
        y_train.append([0, 0, 1, 0])
    for file in os.listdir(pathTrain + "tacos/"):

        #selectedImage = Image.open(pathTrain + f"tacos/{file}").resize(target_resolution).convert('RGB')
        selectedImage = cv2.imread(pathTrain + f"tacos/{file}")
        resized_image = cv2.resize(selectedImage, target_resolution)
        masque = color_treatment(resized_image) / 255

        Ximgs.append(
            np.array(masque))
        y_train.append([0, 0, 0, 1])

    #====================================#
    #===============TEST=================#
    #====================================#


    Ximgs_test = []
    y_test = []
    for file in os.listdir(pathTest+"hotdog/"):

        #selectedImage = Image.open(pathTrain + f"hotdog/{file}").resize(target_resolution).convert('RGB')
        selectedImage = cv2.imread(pathTrain + f"hotdog/{file}")
        resized_image = cv2.resize(selectedImage, target_resolution)
        masque = color_treatment(resized_image) / 255.0

        Ximgs_test.append(
            np.array(masque))
        y_test.append([1, 0, 0, 0])
    for file in os.listdir(pathTest+"burger/"):

        #selectedImage = Image.open(pathTrain + f"burger/{file}").resize(target_resolution).convert('RGB')
        selectedImage = cv2.imread(pathTrain + f"burger/{file}")
        resized_image = cv2.resize(selectedImage, target_resolution)
        masque = color_treatment(resized_image) / 255.0

        Ximgs_test.append(
            np.array(masque))
        y_test.append([0, 1, 0, 0])
    for file in os.listdir(pathTest+"tacos/"):

        #selectedImage = Image.open(pathTrain + f"tacos/{file}").resize(target_resolution).convert('RGB')
        selectedImage = cv2.imread(pathTrain + f"tacos/{file}")
        resized_image = cv2.resize(selectedImage, target_resolution)
        masque = color_treatment(resized_image) / 255.0

        Ximgs_test.append(
            np.array(masque))
        y_test.append([0, 0, 1, 0])
    for file in os.listdir(pathTest + "pizza/"):

        #selectedImage = Image.open(pathTrain + f"pizza/{file}").resize(target_resolution).convert('RGB')
        selectedImage = cv2.imread(pathTrain + f"pizza/{file}")
        resized_image = cv2.resize(selectedImage, target_resolution)
        masque = color_treatment(resized_image) / 255.0

        Ximgs_test.append(
            np.array(masque))
        y_test.append([0, 0, 0, 1])

    x_train = np.array(Ximgs)
    y_train = np.array(y_train)
    x_test = np.array(Ximgs_test)
    y_test = np.array(y_test)
    return (x_train, y_train), (x_test, y_test)



def create_linear_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(4, activation=sigmoid))
    model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['accuracy'])
    return model



def create_mlp_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation=tanh))
    model.add(Dense(256, activation=tanh))
    model.add(Dense(4, activation=sigmoid))
    model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['accuracy'])
    return model


def create_conv_nn_model():
    model = Sequential()

    model.add(Conv2D(4, (3, 3), padding='same', activation=relu))
    model.add(MaxPool2D((2, 2)))

    model.add(Conv2D(8, (3, 3), padding='same', activation=relu))
    model.add(MaxPool2D((2, 2)))

    model.add(Conv2D(16, (3, 3), padding='same', activation=relu))
    # model.add(MaxPool2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation=tanh))
    model.add(Dense(3, activation=sigmoid))
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
    next_tensor = Dense(3, activation=sigmoid)(next_tensor)

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

    next_tensor = Dense(3, activation=sigmoid)(previous_tensor)

    model = keras.models.Model(input_tensor, next_tensor)

    model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['accuracy'])
    return model

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = load_dataset()

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # For conventional

    #model = create_linear_model()
    #model = create_mlp_model()
    #model = create_conv_nn_model()
    model = create_dense_res_nn_model()
    model = create_dense_u_nn_model()

    true_values = np.argmax(y_train, axis=1)
    preds = np.argmax(model.predict(x_train), axis= 1)

    print("Confusion Train Matrix Before Training")
    print(confusion_matrix(true_values, preds))

    true_values = np.argmax(y_test, axis=1)
    preds = np.argmax(model.predict(x_test), axis=1)
    print("Confusion Test Matrix Before Training")
    print(confusion_matrix(true_values, preds))

    print(f'Train Acc : {model.evaluate(x_train, y_train)[1]}')
    print(f'Test Acc : {model.evaluate(x_test, y_test)[1]}')

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


    logs = model.fit(x_train, y_train, batch_size=16, epochs=20, verbose=0, validation_data=(x_test, y_test),
                     callbacks=tensorboard_callback)

    true_values = np.argmax(y_train, axis=1)
    preds = np.argmax(model.predict(x_train), axis=1)

    print("Confusion Train Matrix After Training")
    print(confusion_matrix(true_values, preds))

    true_values = np.argmax(y_test, axis=1)
    preds = np.argmax(model.predict(x_test), axis=1)
    print("Confusion Test Matrix After Training")
    print(confusion_matrix(true_values, preds))
    print(f'Train Acc : {model.evaluate(x_train, y_train)[1]}')
    print(f'Test Acc : {model.evaluate(x_test, y_test)[1]}')

    print(logs.history.keys())

    #Saving The Model
    model.save('linearmodel.keras')
    #new_model = load_model("linearmodel.keras");

    #new_model.summary();


    #plt.show()







    #fig, (ax0, ax1) = plt.subplots(nrows=2, constrained_layout=True)

    #Graph Accuracy
    #ax0.set_title("Accuracy")
    #ax0.plot(logs.history['accuracy'])
    #ax0.plot(logs.history['val_accuracy'])

    #Graph Loss
    #ax1.set_title("Loss")
    #ax1.plot(logs.history['loss'])
    #ax1.plot(logs.history['val_loss'])


    plt.suptitle('Graph Linear Model')
    plt.show()


