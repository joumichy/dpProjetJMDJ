
from modelGenerator import *


#========================MODEL PARAMETER=====================
target_resolution = (64, 64)
numberOfType = 4
epochs = 500
#============================================================

#========================MODELS NAME=========================
TAG_MLP_AUGMENTOR = "mlpmodelAugemntor.keras"
TAG_LINEAR_AUGMENTOR = "linearmodelAugemntor.keras"
TAG_DENSE_RES_NN_AUGMENTOR = "denseresnnmodelAugemntor.keras"
TAG_DENSE_U_NN_AUGMENTOR = "denseunnmodelAugemntor.keras"
TAG_CNN_AUGMENTOR = "cnnAugemntor.keras"
#============================================================


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

    logs  = model.fit_generator(it_train, validation_data= (x_test, y_test), validation_steps= len(x_test)/16,
                    steps_per_epoch=len(x_train) / 16, epochs=epochs,callbacks=tensorboard_callback)

    for e in range(epochs):
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in it_train:
            model.fit(x_batch, y_batch,verbose= 0)
            batches += 1
            if batches >= len(x_train) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
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
