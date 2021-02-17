import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import metrics
from __future__ import division


from google.colab import drive
drive.mount('/content/drive', force_remount=True)

%cd '/content/drive/MyDrive/Colab Notebooks/DeepFake Detector/Celeb-DF-v2'

IMG_HEIGHT, IMG_WIDTH = 155, 155
EPOCHS = 30
DATA_GENERATOR_SEED = 0
LEARNING_RATE = 1e-4
BACTH_SIZE = 32

ROOT_DIR = './data_avg/train/Celeb-synthesis'
TRAIN_DIR = f'{ROOT_DIR}/train'
VALIDATION_DIR = f'{ROOT_DIR}/validation'

TRAIN_DATAGEN = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

TRAIN_GENERATOR = TRAIN_DATAGEN.flow_from_directory(TRAIN_DIR,
                                                    batch_size = BACTH_SIZE,
                                                    class_mode = 'binary', 
                                                    target_size = (IMG_HEIGHT, IMG_WIDTH))

TEST_DATAGEN = ImageDataGenerator(rescale = 1.0/255.)

VALIDATION_GENERATOR = TEST_DATAGEN.flow_from_directory(VALIDATION_DIR,
                                                        batch_size = BACTH_SIZE,
                                                        class_mode = 'binary', 
                                                        target_size = (IMG_HEIGHT, IMG_WIDTH))

print(f'\ntrain real samples={list(TRAIN_GENERATOR.classes).count(0)}\nfake samples={list(TRAIN_GENERATOR.classes).count(1)}')
print(f'\nvalidation real samples={list(VALIDATION_GENERATOR.classes).count(0)}\nfake samples={list(VALIDATION_GENERATOR.classes).count(1)}')

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

MODEL.compile(optimizer = RMSprop(lr = LEARNING_RATE), 
              loss = 'binary_crossentropy', 
              metrics = [
                         metrics.BinaryAccuracy(name = 'acc'),
                         metrics.AUC(name = 'auc'),
                         metrics.FalsePositives(name = 'fp')
                         ])

EARLY_STOPPING = EarlyStopping(monitor='fp', 
                               verbose=1,
                               patience=50,
                               mode='max',
                               restore_best_weights=True)

CLASS_WEIGHT = {0: 0.001694915254237288, 
                1: 0.00017733640716439085}
# CLASS_WEIGHT = {0: 9, 
#                 1: 1}
# CLASS_WEIGHT = {0: 1, 
#                 1: 1}

history = MODEL.fit(TRAIN_GENERATOR,
                    validation_data = VALIDATION_GENERATOR,
                    epochs = EPOCHS,
                    verbose = 2,
                    callbacks = [EARLY_STOPPING],
                    class_weight = CLASS_WEIGHT)

target_names = ['REAL', 'FAKE']
Y_pred = MODEL.predict(VALIDATION_GENERATOR)
y_pred = np.rint(Y_pred)
print(f'CONFUSION MATRIX\n{confusion_matrix(VALIDATION_GENERATOR.classes, y_pred)}')
print(f'\nCLASSIFICATION REPORT\n{classification_report(VALIDATION_GENERATOR.classes, y_pred, target_names=target_names)}')

epochs=range(len(history.history['loss']))

loss_metrics = list(history.history.keys())
for loss in loss_metrics:
    plt.figure()
    plt.plot(epochs, history.history[loss], label=loss)
    plt.title(loss)
    plt.legend()
    plt.show()
