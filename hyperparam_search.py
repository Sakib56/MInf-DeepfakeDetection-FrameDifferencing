from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

import keras
from keras import preprocessing
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import *
from keras.applications import *
from keras import metrics
from keras.losses import BinaryCrossentropy
from keras import backend as K

IMG_HEIGHT, IMG_WIDTH = 224, 224
EPOCHS = 30
DATA_GENERATOR_SEED = 420
LEARNING_RATE = 5e-3
BACTH_SIZE = 32
VALIDATION_SPLIT = 0.1
DF_TYPEZ = ['rnd',
            'avg', 
            'diff',
            'diff-bal',
            'avg-bal',
            'rnd-bal']
LOSSEZ = [BinaryCrossentropy(from_logits=False),
          BinaryCrossentropy(from_logits=True),
          SigmoidFocalCrossEntropy(alpha = 0.25, gamma = 1.0),
          SigmoidFocalCrossEntropy(alpha = 0.25, gamma = 1.5),
          SigmoidFocalCrossEntropy(alpha = 0.25, gamma = 2.0),
          SigmoidFocalCrossEntropy(alpha = 0.25, gamma = 2.5),
          SigmoidFocalCrossEntropy(alpha = 0.25, gamma = 3.0)]

for DF_TYPE in DF_TYPEZ:
    for LOSS in LOSSEZ:
        if 'focal' in LOSS.name:
            EXPERIMENT_NAME = f'{DF_TYPE}-{LOSS.name}-a{LOSS.alpha}-g{LOSS.gamma}'.replace('-', '_')
        else:
            EXPERIMENT_NAME = f'{DF_TYPE}-{LOSS.name}-logits-{LOSS.from_logits}'.replace('-', '_')

        TRAIN_VAL_DIR = f'./Celeb-DF-v2/Celeb-{DF_TYPE}'
        if 'bal' in TRAIN_VAL_DIR:
            TEST_DIR = TRAIN_VAL_DIR.replace('bal', 'test')
        else:
            TEST_DIR = f'{TRAIN_VAL_DIR}-test'

        TRAIN_DATAGEN = ImageDataGenerator(rescale = 1./255.,
                                        rotation_range = 40,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        shear_range = 0.2,
                                        zoom_range = 0.2,
                                        horizontal_flip = True,
                                        validation_split = VALIDATION_SPLIT)

        VAL_DATAGEN = ImageDataGenerator(rescale = 1.0/255., 
                                        validation_split = VALIDATION_SPLIT)

        TEST_DATAGEN = ImageDataGenerator(rescale = 1.0/255.)

        TRAIN_GENERATOR = TRAIN_DATAGEN.flow_from_directory(directory = TRAIN_VAL_DIR,
                                                            batch_size = BACTH_SIZE,
                                                            class_mode = 'binary', 
                                                            target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                            subset = 'training',
                                                            seed = DATA_GENERATOR_SEED)

        VALIDATION_GENERATOR = TRAIN_DATAGEN.flow_from_directory(directory = TRAIN_VAL_DIR,
                                                            batch_size = BACTH_SIZE,
                                                            class_mode = 'binary', 
                                                            target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                            subset = 'validation',
                                                            seed = DATA_GENERATOR_SEED)

        TEST_GENERATOR = TEST_DATAGEN.flow_from_directory(directory = TEST_DIR,
                                                        batch_size = BACTH_SIZE,
                                                        class_mode = 'binary', 
                                                        target_size = (IMG_HEIGHT, IMG_WIDTH),                                
                                                        seed = DATA_GENERATOR_SEED)

        train_gen_list = list(TRAIN_GENERATOR.classes)
        val_gen_list = list(VALIDATION_GENERATOR.classes)
        test_gen_list = list(TEST_GENERATOR.classes)

        train_neg, train_pos = train_gen_list.count(0), train_gen_list.count(1)
        val_neg, val_pos = val_gen_list.count(0), val_gen_list.count(1)
        test_neg, test_pos = test_gen_list.count(0), test_gen_list.count(1)

        pos = train_pos + val_pos + test_pos
        neg = train_neg + val_neg + test_neg
        total = pos + neg

        weight_for_0 = (1 / neg)*(total)/2.0 
        weight_for_1 = (1 / pos)*(total)/2.0

        CLASS_WEIGHT = {0: weight_for_0, 1: weight_for_1}

        STEP_SIZE_TRAIN = TRAIN_GENERATOR.n//TRAIN_GENERATOR.batch_size
        STEP_SIZE_VALID = VALIDATION_GENERATOR.n//VALIDATION_GENERATOR.batch_size

        EARLY_STOPPING = EarlyStopping(monitor = 'auc', 
                                      patience = EPOCHS//2,
                                      mode = 'max',
                                      restore_best_weights = True)
        
        REDUCE_LR  = ReduceLROnPlateau(monitor = 'auc', 
                                       patience = EPOCHS//4,
                                       factor = 0.5,
                                       min_lr=1e-6)
        
        MODEL_CHECKPOINT = ModelCheckpoint(filepath = f'./Checkpoints/{EXPERIMENT_NAME}/',
                                           monitor = 'val_auc',
                                           save_best_only = True)
        
        TENSOR_BOARD = TensorBoard(log_dir = './Checkpoints/{EXPERIMENT_NAME}/logs')

        # print(f'\nTRAIN\nReal samples = {train_neg}\nFake samples = {train_pos}')
        # print(f'\nVALIDATION\nReal samples = {val_neg}\nFake samples = {val_pos}')
        # print(f'\nTEST\nReal samples = {test_neg}\nFake samples = {test_pos}')
        # print(f'\nWeight for Real = {weight_for_0:.5f}')
        # print(f'Weight for Fake = {weight_for_1:.5f}')
        # print(f'\nTrain step size = {STEP_SIZE_TRAIN}')
        # print(f'Validation step size = {STEP_SIZE_VALID}')
        # print(f'\nEarly Stop after {EARLY_STOPPING.patience} epochs of patience based on {EARLY_STOPPING.monitor}')
        # print(f'Model Checkpoints saved to {MODEL_CHECKPOINT.filepath} based on {MODEL_CHECKPOINT.monitor}')
        # print(f'Learning-Rate Scheduler will start reducing Learning-Rate by {0.95} after {EPOCHS//2.5} epochs')

        PRE_TRAINED_MODEL = InceptionV3(input_shape = (IMG_HEIGHT,IMG_WIDTH,3),
                                        include_top = False,
                                        weights = 'imagenet')

        for layer in PRE_TRAINED_MODEL.layers:
            layer.trainable = False

        x = layers.Flatten()(PRE_TRAINED_MODEL.output)  
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)                  
        x = layers.Dense(1, activation = 'sigmoid')(x)  

        MODEL = Model(PRE_TRAINED_MODEL.input, x) 

        MODEL.compile(optimizer = Adam(lr = LEARNING_RATE),
                    loss = LOSS,
                    metrics = [metrics.BinaryAccuracy(name = 'acc'),
                            metrics.AUC(name = 'auc'),
                            metrics.FalsePositives(name = 'fp')])

        HISTORY = MODEL.fit(TRAIN_GENERATOR,
                            epochs = EPOCHS,
                            steps_per_epoch = STEP_SIZE_TRAIN,
                            validation_data = VALIDATION_GENERATOR,
                            validation_steps = STEP_SIZE_VALID,
                            verbose = 2,
                            class_weight = CLASS_WEIGHT,
                            callbacks = [EARLY_STOPPING,
                                         REDUCE_LR,
                                         MODEL_CHECKPOINT,
                                         TENSOR_BOARD])

        acc = HISTORY.history['acc']
        auc = HISTORY.history['auc']
        loss = HISTORY.history['loss']
        fp = HISTORY.history['fp']

        val_acc = HISTORY.history['val_acc']
        val_auc = HISTORY.history['auc']
        val_loss = HISTORY.history['val_loss']
        val_fp = HISTORY.history['val_fp']

        epochs = range(len(acc))

        fig, axs = plt.subplots(2, 2, figsize = (20,20))

        axs[0, 0].plot(epochs, acc, 'r', label='Training Binary Accuracy')
        axs[0, 0].plot(epochs, val_acc, 'b', label='Validation Binary Accuracy')
        axs[0, 0].set_title('Training and Validation Binary Accuracy')
        axs[0, 0].legend()

        axs[0, 1].plot(epochs, loss, 'r', label='Training Loss')
        axs[0, 1].plot(epochs, val_loss, 'b', label='Validation Loss')
        axs[0, 1].set_title('Training and Validation Loss')
        axs[0, 1].legend()

        axs[1, 0].plot(epochs, auc, 'r', label='Training AUC')
        axs[1, 0].plot(epochs, val_auc, 'b', label='Validation AUC')
        axs[1, 0].set_title('Training and Validation AUROC')
        axs[1, 0].legend()

        axs[1, 1].plot(epochs, fp, 'r', label='Training False Positives')
        axs[1, 1].plot(epochs, val_fp, 'b', label='Validation False Positives')
        axs[1, 1].set_title('Training and Validation False Positives')
        axs[1, 1].legend()

        fig.savefig(f'./Checkpoints/{EXPERIMENT_NAME}/plt.png')

        Y_pred = MODEL.predict(TEST_GENERATOR)
        Y_true = TEST_GENERATOR.classes

        text_file = open(f'./Checkpoints/{EXPERIMENT_NAME}/summary.txt', 'w')

        div = 20
        for thrshld in map(lambda x: x/div, range(0,div+1)):
            y_pred = (Y_pred > thrshld).astype(int)
            print(f'THRESHOLD = {thrshld}')
            print(f'\nCONFUSION MATRIX\n{confusion_matrix(Y_true, y_pred)}')
            print(f'\nCLASSIFICATION REPORT\n{classification_report(Y_true, y_pred, target_names = ["REAL", "FAKE"])}\n\n')
            print('________________________________________________________________')
            text_file.write(f'THRESHOLD = {thrshld}\nCONFUSION MATRIX\n{confusion_matrix(Y_true, y_pred)}\nCLASSIFICATION REPORT\n{classification_report(Y_true, y_pred, target_names = ["REAL", "FAKE"])}\n\n')
            
        text_file.write(str(HISTORY.history).replace(', \'', ':\n\n'))
        text_file.close()