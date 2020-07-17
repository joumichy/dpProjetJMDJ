from fileImageLoader import  selectFile
from modelGenerator import  TAG_MLP,\
    TAG_DENSE_RES_NN,TAG_DENSE_U_NN,\
    TAG_CNN, \
    TAG_LINEAR
from tensorflow.keras.models import  load_model
import  cv2
import time
import numpy as np
import tensorflow.keras.models
from modelGeneratorAugmented import TAG_DENSE_U_NN_AUGMENTOR,\
    TAG_LINEAR_AUGMENTOR,\
    TAG_MLP_AUGMENTOR,\
    TAG_CNN_AUGMENTOR,\
    TAG_DENSE_RES_NN_AUGMENTOR


def accueil():

    print("===========================")
    print("==========BIENVENUE=========")
    print("===========================\n")

def print_result(pred_HotDog, pred_Burger, pred_Pizza, pred_Tacos):
    print("RESULTATS DES PREDICTIONS : \n")
    print(f"Probabilitée HotDog  : {pred_HotDog} \n")
    print(f"Probabilitée Burger  : {pred_Burger} \n")
    print(f"Probabilitée Pizza  : {pred_Pizza} \n")
    print(f"Probabilitée Tacos  : {pred_Tacos} \n")


if __name__ == '__main__':

    resolution = (64,64)
    #=========INSET TAG =========
    model_name = TAG_MLP
    #===========================
    model = load_model(f"./models/{model_name}")
    model.summary()
    accueil()

    imageFile = selectFile()
    image = cv2.imread(imageFile)

    try:
        image = cv2.resize(image, resolution)

    except Exception as e:
        print("Erreur lors du traitement (resize) de l'image")

    predictions = model.predict(np.expand_dims(image, axis=0))

    prediction_Hotdog, prediction_Burger, prediction_Pizza, prediction_Tacos  = predictions[0]


    print(predictions)

    print_result(prediction_Hotdog,
                 prediction_Burger,
                 prediction_Pizza,
                 prediction_Tacos)


