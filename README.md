# Convolutional Autoencoder (CAE) in Python

An implementation of a convolutional autoencoder in python and keras.

`cae.py` contains the implementation, which is tested on the MNIST dataset in `mnist_test.ipynb`.

In general, auto-encoders map an input x to a latent representation y (generally in a much smaller dimensional space), using deterministic functions of the type y = sigma(Wx+b). In order to encode images, it is useful to implement a convolutional architecture. Here, we utilize convolutional layers and max-pooling layers (which allow translation-invariant representations), followed by a flattening and dense layer to encode the images in a reduced-dimensional space. For decoding, you essentially need to perform the inverse operation. For more information on CAEs, consult e.g. http://people.idsia.ch/~ciresan/data/icann2011.pdf.
