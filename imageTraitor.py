import cv2


def showImage(masque):
     window_name = "image"
     cv2.imshow(window_name, masque)
     cv2.waitKey(0)
     cv2.destroyWindow(window_name)


def color_treatment(image):
    """Analyse une image et trouve le centre de la forme si elle existe.
        Prend en parametre :
        - image : l'image a analyser

        Renvoie : Une image binaire, composé uniquement de noir et de blanc"""

    ####### Les prochaines lignes permettent de travailler l'image pour #######
    #######            la rendre plus facilement analysable             #######

    # On transforme l'image en niveaux de gris

    gris = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # On floute l'image
    flou = cv2.GaussianBlur(gris, (3, 3), 1)

    # On transforme en image binaire
    # ret, binaire = cv2.threshold(flou, 140, 255, cv2.THRESH_BINARY_INV)

    # On elimine les bruits restants
    # masque = cv2.erode(binaire, None, iterations=2)
    # masque = cv2.dilate(masque, None, iterations=2)

    masque = flou
    masque = cv2.cvtColor(masque, cv2.COLOR_GRAY2RGB)

    #showImage(masque)

    ###########################################################################
    ###########################################################################
    return masque

