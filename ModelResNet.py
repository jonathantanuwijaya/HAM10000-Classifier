import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


def create_model():
    resnet = tf.keras.applications.ResNet152(
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
    return model