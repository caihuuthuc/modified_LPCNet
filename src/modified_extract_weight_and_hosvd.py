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

model.load_weights('/content/drive/MyDrive/checkpoint_original_lpcnet_maybepodcast/maybepodcast-lpcnet20_384_10_G16_05.h5')

# names = [weight.name for layer in model.layers for weight in layer.weights]
# print(names)

kernel_name = 'dual_fc/kernel:0'
bias_name   = 'dual_fc/bias:0'
factor_name = 'dual_fc/factor:0'

# print(dir(model.get_layer("dual_fc")))
kernel_weight = get_weights_by_name(model, kernel_name).numpy()
bias_weight = get_weights_by_name(model, bias_name).numpy()
factor_weight = get_weights_by_name(model, factor_name).numpy()

tucker_tensor = tucker(tl.tensor(kernel_weight), rank=[11, 11, 2])

with open('/content/drive/MyDrive/kernel_weight_of_dualfc.npy', 'wb') as f:
    np.save(f, kernel_weight)
    

reconstructed = tl.tucker_to_tensor(tucker_tensor)

core = tl.to_numpy(tucker_tensor[0])
factor_0 = tl.to_numpy(tucker_tensor[1][0])
factor_1 = tl.to_numpy(tucker_tensor[1][1])
factor_2 = tl.to_numpy(tucker_tensor[1][2])

with open('/content/drive/MyDrive/core_kernel_weight_of_dualfc.npy', 'wb') as f:
    np.save(f, core)

with open('/content/drive/MyDrive/factor_0_kernel_weight_of_dualfc.npy', 'wb') as f:
    np.save(f, factor_0)

with open('/content/drive/MyDrive/factor_1_kernel_weight_of_dualfc.npy', 'wb') as f:
    np.save(f, factor_1)

with open('/content/drive/MyDrive/factor_2_kernel_weight_of_dualfc.npy', 'wb') as f:
    np.save(f, factor_2)

with open('/content/drive/MyDrive/reconstructed_kernel_weight_of_dualfc.npy', 'wb') as f:
    np.save(f, reconstructed)

rec_error = tl.norm(reconstructed - kernel_weight)/tl.norm(kernel_weight)
print("reconstruct error:  ", rec_error)
