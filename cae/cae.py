"""
An implementation of a convolutional autoencoder (CAE) using Keras.

Jason M. Manley, 2018
jmanley@rockefeller.edu
"""
import os
import keras
import numpy as np

class ConvAutoEncoder:

    def __init__(self, input_shape, output_dim, filters=[32, 64, 128, 256],
                 kernel=(3,3), stride=(1,1), strideundo=2, pool=(2,2),
                 optimizer="adamax", lossfn="mse"):
        # For now, assuming input_shape is mxnxc, and m,n are multiples of 2.

        self.input_shape = input_shape
        self.output_dim  = output_dim

        # define encoder architecture
        self.encoder = keras.models.Sequential()
        self.encoder.add(keras.layers.InputLayer(input_shape))
        for i in range(len(filters)):
            self.encoder.add(keras.layers.Conv2D(filters=filters[i], kernel_size=kernel, strides=stride, activation='elu', padding='same'))
            self.encoder.add(keras.layers.MaxPooling2D(pool_size=pool))
        self.encoder.add(keras.layers.Flatten())
        self.encoder.add(keras.layers.Dense(output_dim))

        # define decoder architecture
        self.decoder = keras.models.Sequential()
        self.decoder.add(keras.layers.InputLayer((output_dim,)))
        self.decoder.add(keras.layers.Dense(filters[len(filters)-1] * int(input_shape[0]/(2**(len(filters)))) * int(input_shape[1]/(2**(len(filters))))))
        self.decoder.add(keras.layers.Reshape((int(input_shape[0]/(2**(len(filters)))),int(input_shape[1]/(2**(len(filters)))), filters[len(filters)-1])))
        for i in range(1,len(filters)):
            self.decoder.add(keras.layers.Conv2DTranspose(filters=filters[len(filters)-i], kernel_size=kernel, strides=strideundo, activation='elu', padding='same'))
        self.decoder.add(    keras.layers.Conv2DTranspose(filters=input_shape[2],          kernel_size=kernel, strides=strideundo, activation=None,  padding='same'))

        # compile model
        input         = keras.layers.Input(input_shape)
        code          = self.encoder(input)
        reconstructed = self.decoder(code)

        self.ae = keras.models.Model(inputs=input, outputs=reconstructed)
        self.ae.compile(optimizer=optimizer, loss=lossfn)


    def fit(self, x, epochs=25, callbacks=[keras.callbacks.BaseLogger()], **kwargs):

        self.ae.fit(x=x, y=y, epochs=epochs, callbacks=callbacks, **kwargs)


    def save_weights(self, path=None, prefix=""):
        if path is None: path = os.getcwd()
        self.encoder.save_weights(os.path.join(path, prefix + "encoder_weights.h5"))
        self.decoder.save_weights(os.path.join(path, prefix + "decoder_weights.h5"))


    def load_weights(self, path=None, prefix=""):
        if path is None: path = os.getcwd()
        self.encoder.load_weights(os.path.join(path, prefix + "encoder_weights.h5"))
        self.decoder.load_weights(os.path.join(path, prefix + "decoder_weights.h5"))


    def encode(self, input):
        return self.encoder.predict(input)


    def decode(self, codes):
        return self.decoder.predict(codes)
