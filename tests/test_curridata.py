from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from unittest import TestCase
import numpy as np

from shapeset.curridata import Curridata
from shapeset.polygongen import Polygongen
import shapeset.buildfeaturespolygon as bfp

from keras import models
from keras import layers
from keras import regularizers
from keras import callbacks
from keras import optimizers
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.utils.np_utils import to_categorical
from keras import backend as K

import pygame


class TestCurridata(TestCase):
    def __init__(self, *args, **kwargs):
        if 'main' in kwargs:
            return
        else:
            super(TestCurridata, self).__init__(*args, **kwargs)

    def test_curridata_generator(self):

        np.random.seed(1337)
        image_shape = (32, 32)
        num_classes = 9
        batch_size = 100
        n_epochs = 8000
        n_train_batches = 31
        final_output_size = num_classes

        # the function that uses the data from polygon generator to make outputs (batch_size, num_classes)
        def output_as_categorical(rval_poly_id, n_vertices, nb_poly_max, batchsize, **dic):
            def convertout(out):
                target = 0 * ((out[:, 0] == 1) * (out[:, 1] == 0) * (out[:, 2] == 0)) + \
                         1 * ((out[:, 0] == 0) * (out[:, 1] == 1) * (out[:, 2] == 0)) + \
                         2 * ((out[:, 0] == 0) * (out[:, 1] == 0) * (out[:, 2] == 1)) + \
                         3 * ((out[:, 0] == 1) * (out[:, 1] == 1) * (out[:, 2] == 0)) + \
                         4 * ((out[:, 0] == 0) * (out[:, 1] == 1) * (out[:, 2] == 1)) + \
                         5 * ((out[:, 0] == 1) * (out[:, 1] == 0) * (out[:, 2] == 1)) + \
                         6 * ((out[:, 0] == 2) * (out[:, 1] == 0) * (out[:, 2] == 0)) + \
                         7 * ((out[:, 0] == 0) * (out[:, 1] == 2) * (out[:, 2] == 0)) + \
                         8 * ((out[:, 0] == 0) * (out[:, 1] == 0) * (out[:, 2] == 2))
                return target

            rval_output = np.zeros((batchsize, len(n_vertices)), dtype='int')
            tmp = np.ones(nb_poly_max, dtype='uint8')
            for j in range(batchsize):
                for i in range(len(n_vertices)):
                    rval_output[j, i] = ((rval_poly_id[j, :] == tmp * i).sum())

            rval_integers = convertout(rval_output * 1.0)
            return to_categorical(rval_integers, num_classes)

        # make the Curridata generator
        genparams = {'inv_chance': 0.5, 'img_shape': image_shape, 'n_vert_list': [3, 4, 20], 'fg_min': 0.55, 'fg_max': 1.0,
                     'bg_min': 0.0, 'bg_max': 0.45, 'rot_min': 0.0, 'rot_max': 1.0, 'pos_min': 0, 'pos_max': 1,
                     'scale_min': 0.2, 'scale_max': 0.8, 'rotation_resolution': 255,
                     'nb_poly_max': 1, 'nb_poly_min': 1, 'overlap_max': 0.5, 'poly_type': 1, 'rejectionmax': 50,
                     'overlap_bool': True}
        datagenerator = Polygongen
        funclist = [buildimage_4D, output_as_categorical]
        dependencies = [None, None]
        funcparams = {'neighbor': 'V8', 'gaussfiltbool': False, 'sigma': 0.5, 'size': 5, 'neg': True}
        seed = 0
        curridata = Curridata(datagenerator, genparams, funclist, dependencies, funcparams, batch_size, seed,
                              generatorReturnsBatch=True, feature_input=0, feature_output=1)

        # build the architecture for the ANN

        # dense architecture
        inputTensor = layers.Input(shape=(image_shape[0] * image_shape[1], ), name="Inputs")
        lastOutputTensor = inputTensor
        activation = 'tanh'
        lastOutputTensor = layers.Dense(1000, activation=activation)(lastOutputTensor)
        lastOutputTensor = layers.Dense(1000, activation=activation)(lastOutputTensor)
        lastOutputTensor = layers.Dense(1000, activation=activation)(lastOutputTensor)
        lastOutputTensor = layers.Dense(1000, activation=activation)(lastOutputTensor)
        lastOutputTensor = layers.Dense(1000, activation=activation)(lastOutputTensor)

        # # convolutional architecture
        # inputTensor = layers.Input(shape=image_shape + (1,), name="Inputs")
        # lastOutputTensor = inputTensor
        # lastOutputTensor = layers.Conv2D(filters=8, kernel_size=3, activation='relu')(lastOutputTensor)
        # lastOutputTensor = layers.MaxPooling2D(pool_size=2)(lastOutputTensor)
        # lastOutputTensor = layers.Conv2D(filters=4, kernel_size=(3, 3), activation='relu', padding='same')(lastOutputTensor)
        # lastOutputTensor = layers.GlobalAveragePooling2D()(lastOutputTensor)

        # final softmax layer
        lastOutputTensor = layers.Dense(final_output_size, activation='linear')(lastOutputTensor)
        lastOutputTensor = layers.Activation(activation='softmax')(lastOutputTensor)
        model = models.Model(inputs=inputTensor, outputs=lastOutputTensor)

        # compile the model
        metrics = ['categorical_accuracy']
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=metrics)
        model.summary()

        # fit the model
        callbacks = []
        history = model.fit_generator(generator=curridata,
                                      steps_per_epoch=n_train_batches,
                                      epochs=n_epochs,
                                      verbose=2,
                                      callbacks=callbacks)

        # how did we do?
        self.assertTrue(history.history['categorical_accuracy'][-1] > 0.75)


if __name__ == '__main__':
    tklf = TestCurridata(main=True)
    tklf.setUp()
    tklf.test_curridata_generator()
