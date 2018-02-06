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
from keras.callbacks import ReduceLROnPlateau


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
        batch_size = 10
        n_epochs = 8000
        n_train_batches = 31
        final_output_size = num_classes

        # make the Curridata generator
        genparams = {'inv_chance': 0.5, 'img_shape': image_shape, 'n_vert_list': [3, 4, 20], 'fg_min': 0.55, 'fg_max': 1.0,
                     'bg_min': 0.0, 'bg_max': 0.45, 'rot_min': 0.0, 'rot_max': 1.0, 'pos_min': 0, 'pos_max': 1,
                     'scale_min': 0.2, 'scale_max': 0.8, 'rotation_resolution': 255,
                     'nb_poly_max': 2, 'nb_poly_min': 1, 'overlap_max': 0.5, 'poly_type': 2, 'rejectionmax': 50,
                     'overlap_bool': True}
        datagenerator = Polygongen
        funclist = [bfp.buildimage_4D, bfp.output_as_Shapeset3x2_categorical]
        dependencies = [None, None]
        funcparams = {'neighbor': 'V8', 'gaussfiltbool': False, 'sigma': 0.5, 'size': 5, 'neg': True}
        seed = 0
        curridata = Curridata(datagenerator, genparams, funclist, dependencies, funcparams, batch_size, seed,
                              generatorReturnsBatch=True, feature_input=0, feature_output=1)

        # build the architecture for the ANN

        # # dense architecture. make sure to change funclist first function to buildimage
        # inputTensor = layers.Input(shape=(image_shape[0] * image_shape[1],), name="Inputs")
        # lastOutputTensor = inputTensor
        # activation = 'tanh'
        # lastOutputTensor = layers.Dense(1000, activation=activation)(lastOutputTensor)
        # lastOutputTensor = layers.Dense(1000, activation=activation)(lastOutputTensor)
        # lastOutputTensor = layers.Dense(1000, activation=activation)(lastOutputTensor)
        # lastOutputTensor = layers.Dense(1000, activation=activation)(lastOutputTensor)
        # lastOutputTensor = layers.Dense(1000, activation=activation)(lastOutputTensor)

        # convolutional architecture. make sure to change funclist first function to buildimage_4D
        inputTensor = layers.Input(shape=image_shape + (1,), name="Inputs")
        lastOutputTensor = inputTensor
        lastOutputTensor = layers.Conv2D(filters=8, kernel_size=3, activation='relu')(lastOutputTensor)
        lastOutputTensor = layers.MaxPooling2D(pool_size=2)(lastOutputTensor)
        lastOutputTensor = layers.Conv2D(filters=4, kernel_size=(3, 3), activation='relu', padding='same')(lastOutputTensor)
        lastOutputTensor = layers.GlobalAveragePooling2D()(lastOutputTensor)

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
        training_callbacks = []
        rlr = ReduceLROnPlateau(factor=0.8, patience=100, verbose=1)
        training_callbacks.append(rlr)
        history = model.fit_generator(generator=curridata,
                                      steps_per_epoch=n_train_batches,
                                      epochs=n_epochs,
                                      verbose=2,
                                      callbacks=training_callbacks)

        # how did we do?
        self.assertTrue(history.history['categorical_accuracy'][-1] > 0.50)


if __name__ == '__main__':
    tklf = TestCurridata(main=True)
    tklf.setUp()
    tklf.test_curridata_generator()
