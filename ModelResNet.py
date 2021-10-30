import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers, models
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D


def create_model():
    resnet = tf.keras.applications.ResNet50(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )

    output = resnet.output
    output = Dense(5, activation='softmax', name='SoftMax')(output)
    model = Model(inputs=resnet.input, outputs=output)

    model.summary()
    return model
