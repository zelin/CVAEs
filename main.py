from tensorflow.keras.layers import Dense, Input, Conv2D, Conv2DTranspose, MaxPooling2D, LeakyReLU, Reshape, BatchNormalization, Flatten, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
import numpy as np

K.clear_session()

def loadData():
    # y_train contains the labels, numbers are 1, 2 or 5, 7 etc.

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalizing based on color 255
    x_train = x_train.astype(np.float32) /255.0
    x_test = x_test.astype(np.float32)/255.0
    # convert shape of x_train from (60000, 28, 28) to (60000, 784)
    # 784 columns per row
    # x_train = x_train.reshape(60000, 784)
    return (np.expand_dims(x_train, axis=-1), y_train, np.expand_dims(x_test, axis=-1), y_test)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend

def sampling(args):

    # Arguments
    #    args (tensor): mean and log of variance of Q(z|X)
    # Returns
    #    z (tensor): sampled latent vector
    # Taken from StackOverFlow
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    # epsilon is stochastic.
    # Returns a tensor with normal distribution of values.
    epsilon = K.random_normal(shape=(batch, dim))
    # mean + variance * epsilon , converting logVar into variance
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def kl_loss(z_mean, z_log_var):

  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  return K.mean(kl_loss)


def getEncoder():
    # This returns a tensor
    inputs = Input(shape=(28, 28, 1))
    # Making a sample AlexNet Model Layer 1
    # A layer that consists of a set of “filters”. The filters take a subset of the input data at a time, but are
    # applied across the full input (by sweeping over the input). The operations performed by this layer are still
    # linear/matrix multiplications, but they go through an activation function at the output, which is usually a
    # non-linear operation.
    # We are using 32 filters of size 4,4 Filters are choosen by Keras itself
    # Default filters used by Keras
    # https://datascience.stackexchange.com/questions/16463/what-is-are-the-default-filters-used-by-keras-convolution2d
    encoder = Conv2D(32, (4, 4), padding='same', activation='relu')(inputs)
    # A pooling layer effectively down samples the output of the prior layer, reducing the number of operations
    # required for all the following layers, but still passing on the valid information from the previous layer.
    # Max pooling sums up the closeby matrix points and place them in middle.
    encoder = MaxPooling2D((4, 4), strides=(2, 2), padding='same')(encoder) # 14,14
    # Used at the input for feature scaling, and in batch normalisation at hidden layers.
    encoder = BatchNormalization()(encoder)
    # Making a sample AlexNet Model Layer 2
    encoder = Conv2D(64, (2, 2), padding='same', activation='relu')(encoder)
    encoder = MaxPooling2D((4, 4), strides=(2, 2), padding='same')(encoder) # 7, 4
    encoder = BatchNormalization()(encoder)
    # Flatten entire row, in our case the result of batch norm, 7x7x64 = 3136
    encoder = Flatten()(encoder)
    latentDimensions = 64
    # Name is used to directly locate the layer, otherwise keras will name it based on its internal mechanism
    zMeanLayer = Dense(latentDimensions, name='z_mean')(encoder)
    zLogLayer  = Dense(latentDimensions, name='z_log_var')(encoder)
    # Lambda layer is a layer that wraps an arbitrary expression. For example, at a point you want to calculate the
    # square of a variable but you can not only put the expression into you model because it only accepts layer so you
    # need Lambda function to make your expression be a valid layer in Keras.
    # This is to perform the reparametrization trick
    # Arguments are : Function -> Output and given arguments of function in ()
    z = Lambda(sampling, output_shape=(latentDimensions,), name='z')([zMeanLayer, zLogLayer])
    # Creating final encoder --- We have 2 output layers for VAEs and a third for epsilon i-e reparametrization
    encoder = Model(inputs=inputs, outputs=[zMeanLayer, zLogLayer, z])
    encoder.summary()
    return encoder


def getDecoder():

    # This returns a tensor of shape (None, 28, 28, 1) exact same shape as input
    # our lambda function returns a tensor of size None, 64
    latentDimensions = 64
    inputs = Input(shape=(latentDimensions,))
    # We need image of 28,28 --- Nice conversions
    # Output is none,49 7,7,1 for image
    disc = Dense(49)(inputs)
    disc = LeakyReLU(alpha=0.2)(disc)
    disc = Reshape([7,7,1])(disc)
    # a layer instance is callable on a tensor, and returns a tensor
    # 32 Filters
    # Transpose2D Layer is DeConv layer, a good explanation is given on
    # https: // datascience.stackexchange.com / questions / 6107 / what - are - deconvolutional - layers
    disc = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu')(disc)
    disc = Conv2DTranspose(1, (3,3), strides=(2,2), padding='same', activation='sigmoid')(disc)
    decoder = Model(inputs=inputs, outputs=disc)
    decoder.summary()
    return decoder

def createVAE(decoder, encoder):
    # We are saying that the decoder takes the last output of the encoder as the input
    dec_out = decoder(encoder.outputs[2]) 
    # Defining an end-to-end model with encoder inputs and decoder outputs
    vae = Model(inputs=encoder.inputs, outputs=dec_out)
    vae.summary()
    # VAE loss comprises both crossentropy and KL divergence loss
    vae.compile(loss='binary_crossentropy', optimizer='rmsprop')
    vae.add_loss(kl_loss(encoder.outputs[0], encoder.outputs[1]))
    return vae


def doTraining(epochs=1, batchSize=128):
    # Loading the data
    (mniTrainX, mniTrainY, mniTestX, mniTestY) = loadData()
    # Creating GAN
    encoder = getEncoder()
    decoder = getDecoder()
    vae = createVAE(decoder, encoder)

    # I have removed the tensorboard callback. If needed add that.
    vae.fit(mniTrainX, mniTrainX,
                    epochs=epochs,
                    batch_size=batchSize,
                    shuffle=True,
                    validation_data=(mniTestX, mniTestX)
                    )

encoder = getEncoder()
decoder = getDecoder()
vae = createVAE(decoder, encoder)
doTraining()