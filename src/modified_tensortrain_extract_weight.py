#!/usr/bin/python3

import lpcnet
import sys
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from ulaw import ulaw2lin, lin2ulaw
import keras.backend as K
import h5py

import tensorly as tl
from tensorly.decomposition import tucker

import tensorflow as tf
from tensorflow.python.keras.backend import set_session

def get_weights_by_name(model, name):
    return [w for w in model.weights if w.name==name][0]

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.compat.v1.Session(config=config))

model, enc, dec = lpcnet.new_lpcnet_model(use_gpu=False)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

model.load_weights('/content/drive/MyDrive/lpcnet20_384_10_G16_02.h5')

names = [weight.name for layer in model.layers for weight in layer.weights]
for name in names:
  print(name)
