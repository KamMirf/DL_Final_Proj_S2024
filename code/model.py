import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization

import hyperparameters as hp
from keras import losses
from keras import optimizers
from keras import regularizers

"""VGG CNN pretrained on ImageNet"""
class VGGModel(tf.keras.Model):
       
    def __init__(self):
        super(VGGModel, self).__init__()

        self.optimizer = optimizers.Adam(hp.learning_rate)

        self.vgg16 = [
            # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]

        #set VGG layers to untrainable because we use pretrained weights
        for layer in self.vgg16:
               layer.trainable = False

       #custom head to predict real or fake
        self.head = [
            Flatten(),
            Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                  kernel_regularizer=regularizers.l2(0.01)),
            Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                  kernel_regularizer=regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            #Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01)) #2 classes = real or fake
            Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))  # Single output neuron for binary classification

        ]

        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg16(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        return losses.BinaryCrossentropy()(labels, predictions)
