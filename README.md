# Convolutional Autoencoder (CAE) in Python

An implementation of a convolutional autoencoder in python and keras.

## Installation

`pip install cae`

## Usage

```
from cae import cae
import numpy as np

# create a fake dataset, here: 1000 random 224x224 RGB images
images = np.random.normal(size=(1000, 224, 224, 3))

latent_dim = 8 # desired latent dimensionality

model = cae(images.shape[1:], latent_dim) # there are a number of **kwargs
                                          # parameters that are likely
                                          # worth tuning!!!

# TRAIN THE NETWORK
model.fit(images)

# SAVE THE WEIGHTS FOR EASY RELOADING LATER WITH model.load_weights(path)
model.save_weights('PATH/TO/SAVE/')
```

## Final words

`cae.py` contains the implementation, which is tested on the MNIST dataset in `mnist_test.ipynb`.

In general, auto-encoders map an input x to a latent representation y (generally in a much smaller dimensional space), using deterministic functions of the type y = sigma(Wx+b). In order to encode images, it is useful to implement a convolutional architecture. Here, we utilize convolutional layers and max-pooling layers (which allow translation-invariant representations), followed by a flattening and dense layer to encode the images in a reduced-dimensional space. For decoding, you essentially need to perform the inverse operation. For more information on CAEs, consult e.g. http://people.idsia.ch/~ciresan/data/icann2011.pdf.
