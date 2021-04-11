# Imports
from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from numpy.random import seed
from time import time
from pathlib import Path

import tensorflow as tf

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, RMSprop, SGD
from keras.applications import Xception
from keras import metrics
from keras.losses import BinaryCrossentropy
from keras import backend as K

from tensorflow_addons.losses import SigmoidFocalCrossEntropy

from kerastuner.tuners import Hyperband
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters
from kerastuner import Objective

IMG_HEIGHT, IMG_WIDTH = 71, 71

def loss_experiment_model(hp):
    PRE_TRAINED_MODEL = Xception(input_shape = (IMG_HEIGHT,IMG_WIDTH,3),
                                 include_top = False,
                                 pooling = 'avg',
                                 weights = 'imagenet')
    
    for layer in PRE_TRAINED_MODEL.layers:
        layer.trainable = False

    x = layers.Flatten()(PRE_TRAINED_MODEL.output)
    x = layers.Dense(1, activation = 'sigmoid')(x)

    MODEL = Model(PRE_TRAINED_MODEL.input, x)

    chosen_loss = hp.Choice('loss', values=['SigmoidFocalCrossEntropy', 'BinaryCrossentropy'])
    MODEL.compile(optimizer = 'adam',
                loss = eval(chosen_loss)(), 
                metrics = [metrics.BinaryAccuracy(name = 'acc'),
                           metrics.AUC(name = 'auc'),
                           metrics.FalsePositives(name = 'fp')])
    return MODEL

def optimiser_experiment_model(hp):
    PRE_TRAINED_MODEL = Xception(input_shape = (IMG_HEIGHT,IMG_WIDTH,3),
                                 include_top = False,
                                 pooling = 'avg',
                                 weights = 'imagenet')
    
    for layer in PRE_TRAINED_MODEL.layers:
        layer.trainable = False

    x = layers.Flatten()(PRE_TRAINED_MODEL.output)
    x = layers.Dense(1, activation = 'sigmoid')(x)

    MODEL = Model(PRE_TRAINED_MODEL.input, x)

    MODEL.compile(optimizer = hp.Choice('optimiser', values=['Adam', 'RMSprop', 'SGD']),
                loss = 'binary_crossentropy', 
                metrics = [metrics.BinaryAccuracy(name = 'acc'),
                           metrics.AUC(name = 'auc'),
                           metrics.FalsePositives(name = 'fp')])
    return MODEL

def structure_experiment_model(hp):
    PRE_TRAINED_MODEL = Xception(input_shape = (IMG_HEIGHT,IMG_WIDTH,3),
                                 include_top = False,
                                 pooling = 'avg',
                                 weights = 'imagenet')
    
    for layer in PRE_TRAINED_MODEL.layers:
        layer.trainable = False

    x = layers.Flatten()(PRE_TRAINED_MODEL.output)
    for i in range(hp.Int('number_of_dense_dropout_blocks', min_value = 0, max_value = 2)):
        x = layers.Dense(hp.Int(f'dense_units_{i}', min_value = 1024, max_value = 4096, step = 512), activation = 'relu')(x)
        x = layers.Dropout(hp.Float(f'dropout_probability_{i}', min_value = 0, max_value = 0.3, step = 0.05))(x)
    x = layers.Dense(hp.Int(f'penultimate_dense_unit', min_value = 1024, max_value = 4096, step = 512), activation = 'relu')(x)
    x = layers.Dense(1, activation = 'sigmoid')(x)

    MODEL = Model(PRE_TRAINED_MODEL.input, x)

    MODEL.compile(optimizer = 'adam',
                loss = 'binary_crossentropy', 
                metrics = [metrics.BinaryAccuracy(name = 'acc'),
                           metrics.AUC(name = 'auc'),
                           metrics.FalsePositives(name = 'fp')])
    return MODEL

def all_experiment_model(hp):
    PRE_TRAINED_MODEL = Xception(input_shape = (IMG_HEIGHT,IMG_WIDTH,3),
                                 include_top = False,
                                 pooling = 'avg',
                                 weights = 'imagenet')
    
    for layer in PRE_TRAINED_MODEL.layers:
        layer.trainable = False

    x = layers.Flatten()(PRE_TRAINED_MODEL.output)
    for i in range(hp.Int('number_of_dense_dropout_blocks', min_value = 0, max_value = 2)):
        x = layers.Dense(hp.Int(f'dense_units_{i}', min_value = 1024, max_value = 4096, step = 512), activation = 'relu')(x)
        x = layers.Dropout(hp.Float(f'dropout_probability_{i}', min_value = 0, max_value = 0.3, step = 0.05))(x)
    x = layers.Dense(hp.Int(f'penultimate_dense_unit', min_value = 1024, max_value = 4096, step = 512), activation = 'relu')(x)
    x = layers.Dense(1, activation = 'sigmoid')(x)

    MODEL = Model(PRE_TRAINED_MODEL.input, x)

    chosen_loss = hp.Choice('loss', values=['SigmoidFocalCrossEntropy', 'BinaryCrossentropy'])
    MODEL.compile(optimizer = hp.Choice('optimiser', values=['Adam', 'RMSprop', 'SGD']),
                loss = eval(chosen_loss)(), 
                metrics = [metrics.BinaryAccuracy(name = 'acc'),
                           metrics.AUC(name = 'auc'),
                           metrics.FalsePositives(name = 'fp')])
    return MODEL