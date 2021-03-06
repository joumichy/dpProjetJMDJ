# Fast Food Detector

## Summary
Ce présent document représente le Readme du front de l'application du groupe 7 4AL projet de Deep Learning 2019-2020.
Cette application permet de réaliser les différents Model proposé dans le fichier "modelGenerator.py" et "modelGeneratorAugmented.py"
Ainsi que de détecter des plats de fast foods suite à une image insérrée par l'utilisateur.

## Traitement d'image

Noter dataset est composé de 4 catégories : 

    - Hotdog
    - Burger
    - Pizza
    - Tacos
    
Nous utilisons les librairies Pillow et plus particulièrement OpenCV afin de traiter les images du dataset. De cette façon nous optimisons la définition de ces classes
en éléminant une partie des bruits présentes dans les images téléchargées.

## Installation

Afin d'utiliser l'application, clonez ce dernier projet en veillant d'avoir tensorflow.
Vous aurez également besoin de la librairie OpenCV.
Enfin, la librairie PILLOW est également nécessaire.

Liens librairies : 

    - TensorFlow https://www.tensorflow.org/install/pip?hl=fr
    - OpenCV     https://pypi.org/project/opencv-python/
    - Pillow     https://pypi.org/project/Pillow

## Utilisation de l'application (Detection d'image)

Modification à effectuer dans le fichier : mainClass.py

Si vous  désirez détecter une image en fonction d'un model en particulier, il vous suffit d'insérer l'un des tag proposé dans le fichier à la ligne 167
à la place du TAG actuel (TAG_MLP)

liste des models proposés :

    - TAG_MLP
    - TAG_DENSE_RES_NN
    - TAG_DENSE_U_NN
    - TAG_CNN
    - TAG_LINEAR
    - TAG_DENSE_RES_NN_AUGMENTOR
    - TAG_CNN_AUGMENTOR
    - TAG_MLP_AUGMENTOR
    - TAG_LINEAR_AUGMENTOR
    - TAG_DENSE_U_NN_AUGMENTOR

 Enfin, il vous sufit ensuite de lancer l'application. (*Commande à exéctuer* : **python mainClass.py**)
 
 ## Utilisation de l'application (Conception d'un model)
 
 Modification à effectuer dans le fichier : modelGenerator.py
 
 Si vous désirez concevoir un model pour ce faire, il vous suffit de retirer l'annotation "#" devant l'une des fonctions proposées entre les lignes 155 à 159.
 Seul un model doit rester sans annotations.
 
 liste des noms des fonctions de créations de model :
 
    - create_linear_model()
    - create_mlp_model()
    - create_conv_nn_model()
    - create_dense_res_nn_model()
    - create_dense_u_nn_model()
 
 Ensuite insérer le TAG à la place du TAG (actuelle à la ligne 68)
 
 
 liste des models proposés :
 
    - TAG_MLP
    - TAG_DENSE_RES_NN
    - TAG_DENSE_U_NN
    - TAG_CNN
    - TAG_LINEAR
 
 il vous sufit ensuite de lancer l'application. (*Commande à exéctuer* : **python modelGenerator.py**)
 
  ## Utilisation de l'application (Conception d'un model de type dit "Augmenté")
 
 Modification à effectuer dans le fichier : modelGenerator.py
 
 Ces models sont entrainées de manières différentes des autres models proposées
 
 Si vous désirez concevoir un model pour ce faire, il vous suffit de retirer l'annotation "#" devant l'une des fonctions proposées entre les lignes 155 à 159.
 Seul un model doit rester sans annotations.
 
 Nom des fonctions : (cf Conception d'un model)
 
 Ensuite insérer le TAG à la place du TAG (actuelle à la ligne 190)
 
 liste des models augmentés proposés :
 
    - TAG_DENSE_RES_NN_AUGMENTOR
    - TAG_CNN_AUGMENTOR
    - TAG_MLP_AUGMENTOR
    - TAG_LINEAR_AUGMENTOR
    - TAG_DENSE_U_NN_AUGMENTOR
 
 il vous sufit ensuite de lancer l'application. (*Commande à exéctuer* : **python modelGenerator.py**)
 
 
 Une fois votre model generé, un graphique s'affichera avec des courbes représentants l'accuracy et les loss pour le model entrainer ainsi que pour ses tests.
 
 ## Visualisation Tensorboard
 
 *Commande à exéctuer* : **tensorboard.bat**
 
 Vous pouvez également visualiser en local les statistiques de vos models en éxécutant la commande ci dessus, vous pouvez vous rendre au lien :  http://localhost:6006/#scalars afin de consulter
 vos différents models entrainées. 
 Cet outil propose une visualisation plus avancées de la conceptions de vos models.


 Enjoy!
 
 Auteur : Groupe 7 ESGI 4AL1 2019-2020 (ALLOU John, GHALEM Marc, TRAORE Djadji, CHATELIN Joseph).
