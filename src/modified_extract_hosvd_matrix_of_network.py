#!/usr/bin/python3

import modified_lpcnet as lpcnet
import sys
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from ulaw import ulaw2lin, lin2ulaw
import keras.backend as K
import h5py

import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.decomposition import partial_tucker

import tensorflow as tf
from tensorflow.python.keras.backend import set_session

def get_weights_by_name(model, name):
    return [w for w in model.weights if w.name==name][0]

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.compat.v1.Session(config=config))

model, enc, dec = lpcnet.new_lpcnet_model(use_gpu=False)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

model.load_weights('/content/drive/MyDrive/checkpoint_hosvd_lpcnet/hosvd_mdense_lpcnet20_384_10_G16_01.h5', by_name=True,) 

# names = [weight.name for layer in model.layers for weight in layer.weights]
# print(names)

core_name = 'dual_fc/hosvd_core:0'
factor_0_name = 'dual_fc/hosvd_factor_0:0'
factor_1_name = 'dual_fc/hosvd_factor_1:0'
factor_2_name = 'dual_fc/hosvd_factor_2:0'


# print(dir(model.get_layer("dual_fc")))
core_weight = get_weights_by_name(model, core_name).numpy()
factor_0_weight = get_weights_by_name(model, factor_0_name).numpy()
factor_1_weight = get_weights_by_name(model, factor_1_name).numpy()
factor_2_weight = get_weights_by_name(model, factor_2_name).numpy()


reconstructed = tl.tucker_to_tensor(tucker_tensor)

core = tl.to_numpy(tucker_tensor[0])
factor_0 = tl.to_numpy(tucker_tensor[1][0])
factor_1 = tl.to_numpy(tucker_tensor[1][1])
factor_2 = tl.to_numpy(tucker_tensor[1][2])


with open('/content/drive/MyDrive/kernel_weight_of_dualfc.npy', 'rb') as f:
    kernel_weight = np.load(f)

rec = tenalg.multi_mode_dot(core, [factor_0, factor_1, factor_2])
rec_error = tl.norm(rec - kernel_weight)/tl.norm(kernel_weight)
