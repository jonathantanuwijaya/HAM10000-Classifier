import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers, models
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

def create_model():
    # resnet = tf.keras.applications.ResNet152(
    #     include_top=True,
    #     weights="imagenet",
    #     input_tensor=None,
    #     input_shape=None,
    #     pooling=None,
    #     classes=1000,
    # )
    # output = resnet.output
    # output = Dense(5, activation='softmax', name='SoftMax')(output)
    # model = Model(inputs=resnet.input, outputs=output)
    # return model
    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # # model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(5))
    model = models.Sequential()
    model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())

    model.add(Dense(32))
    model.add(Dense(5, activation='softmax'))
    return model
