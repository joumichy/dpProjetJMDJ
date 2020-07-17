import tkinter as tk
from PIL import Image
from tkinter import filedialog
from imageTraitor import  color_treatment
import cv2
from imageTraitor import  color_treatment
import numpy as np
import os


target_resolution = (64, 64)

def selectFile():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()

    print(f"Image selsected : {file_path} ")
    return(file_path)


def load_dataset():
    Ximgs = []
    y_train = []
    pathTrain = "./dataset/train/"
    pathTest = "./dataset/test/"

    for file in os.listdir(pathTrain + "hotdog/"):
        try:

            selectedImage = cv2.imread(pathTrain + f"hotdog/{file}")
            resized_image = cv2.resize(selectedImage, target_resolution)
            masque = color_treatment(resized_image)

            # image = Image.open(pathTrain+f"hotdog/{file}").resize(target_resolution).convert('RGB')
            # Ximgs.append( np.array(image) / 255.0)

            Ximgs.append(masque / 255.0)
            y_train.append([1, 0, 0, 0])
        except Exception as e:
            pass



    for file in os.listdir(pathTrain + "burger/"):
        try:
            selectedImage = cv2.imread(pathTrain + f"burger/{file}")
            resized_image = cv2.resize(selectedImage, target_resolution)
            masque = color_treatment(resized_image)
            Ximgs.append(masque / 255.0)

            #Ximgs.append(np.array(Image.open(pathTrain+f"burger/{file}").resize(target_resolution).convert('RGB')) / 255.0)
            y_train.append([0, 1, 0, 0])

        except Exception as e:
            pass

    for file in os.listdir(pathTrain + "pizza/"):
        try:
            selectedImage = cv2.imread(pathTrain + f"pizza/{file}")
            resized_image = cv2.resize(selectedImage, target_resolution)
            masque = color_treatment(resized_image)
            Ximgs.append(masque / 255.0)

            # Ximgs.append(np.array(Image.open(pathTrain+f"pizza/{file}").resize(target_resolution).convert('RGB')) / 255.0)
            y_train.append([0, 0, 1, 0])

        except Exception as e:
            pass

    for file in os.listdir(pathTrain + "tacos/"):
        try:
            selectedImage = cv2.imread(pathTrain + f"tacos/{file}")
            resized_image = cv2.resize(selectedImage, target_resolution)
            masque = color_treatment(resized_image)
            Ximgs.append(masque / 255.0)

            #Ximgs.append(np.array(Image.open(pathTrain+f"tacos/{file}").resize(target_resolution).convert('RGB')) / 255.0)
            y_train.append([0, 0, 0, 1])

        except Exception as e:
            pass
    Ximgs_test = []
    y_test = []
    for file in os.listdir(pathTest + "hotdog/"):
        try:
            selectedImage = cv2.imread(pathTest + f"hotdog/{file}")
            resized_image = cv2.resize(selectedImage, target_resolution)
            masque = color_treatment(resized_image)
            Ximgs_test.append(masque / 255.0)

            #Ximgs_test.append(  np.array(Image.open(pathTest+f"hotdog/{file}").resize(target_resolution).convert('RGB')) / 255.0)
            y_test.append([1, 0, 0, 0])

        except Exception as e:
            pass
    for file in os.listdir(pathTest + "burger/"):
        try:
            selectedImage = cv2.imread(pathTest + f"burger/{file}")
            resized_image = cv2.resize(selectedImage, target_resolution)
            masque = color_treatment(resized_image)
            Ximgs_test.append(masque / 255.0)

            #Ximgs_test.append( np.array(Image.open(pathTest+f"burger/{file}").resize(target_resolution).convert('RGB')) / 255.0)
            y_test.append([0, 1, 0, 0])

        except Exception as e:
            pass

    for file in os.listdir(pathTest + "pizza/"):
        try:
            selectedImage = cv2.imread(pathTest + f"pizza/{file}")
            resized_image = cv2.resize(selectedImage, target_resolution)
            masque = color_treatment(resized_image)
            Ximgs_test.append(masque / 255.0)

            #Ximgs_test.append( np.array(Image.open(pathTest+f"pizza/{file}").resize(target_resolution).convert('RGB')) / 255.0)
            y_test.append([0, 0, 1, 0])

        except Exception as e:
            pass

    for file in os.listdir(pathTest + "tacos/"):
        try:
            selectedImage = cv2.imread(pathTest + f"tacos/{file}")
            resized_image = cv2.resize(selectedImage, target_resolution)
            masque = color_treatment(resized_image)
            Ximgs_test.append(masque / 255.0)

            #Ximgs_test.append( np.array(Image.open(pathTest + f"tacos/{file}").resize(target_resolution).convert('RGB')) / 255.0)
            y_test.append([0, 0, 0, 1])

        except Exception as e:
            pass

    x_train = np.array(Ximgs)
    y_train = np.array(y_train)
    x_test = np.array(Ximgs_test)
    y_test = np.array(y_test)
    return (x_train, y_train), (x_test, y_test)

