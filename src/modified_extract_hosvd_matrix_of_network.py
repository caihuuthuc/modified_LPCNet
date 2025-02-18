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
from tensorly import tenalg

import tensorflow as tf
from tensorflow.python.keras.backend import set_session

def get_original_dualfc_weight():
    with open('/content/drive/MyDrive/core_kernel_weight_of_dualfc.npy', 'rb') as f:
        core_weight = np.load(f)

    with open('/content/drive/MyDrive/factor_0_kernel_weight_of_dualfc.npy', 'rb') as f:
        factor_0_weight = np.load(f)

    with open('/content/drive/MyDrive/factor_1_kernel_weight_of_dualfc.npy', 'rb') as f:
        factor_1_weight = np.load(f)

    with open('/content/drive/MyDrive/factor_2_kernel_weight_of_dualfc.npy', 'rb') as f:
        factor_2_weight = np.load(f)

    return tenalg.multi_mode_dot(core_weight, [factor_0_weight, factor_1_weight, factor_2_weight])
            
'''Adapted from dump_lpcnet.py'''
def printVector(vector, name, dtype='float'):
    outline = ""
    v = np.reshape(vector, (-1));
    #print('static const float ', name, '[', len(v), '] = \n', file=f)
    outline += 'static const {} {}[{}] = {{\n   '.format(dtype, name, len(v))
    for i in range(0, len(v)):
        outline += '{}'.format(v[i])
        if (i!=len(v)-1):
            outline += ','
        else:
            break;
        if (i%8==7):
            outline += "\n   "
        else:
            outline += " "
    #print(v, file=f)
    outline += '\n};\n\n'
    return outline
    
def get_weights_by_name(model, name):
    return [w for w in model.weights if w.name==name][0]

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.compat.v1.Session(config=config))

model, enc, dec = lpcnet.new_lpcnet_model(use_gpu=False)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

weight_path = '/content/drive/MyDrive/checkpoint_hosvd_lpcnet/hosvd_mdense_lpcnet20_384_10_G16_03.h5'
model.load_weights(weight_path) 
print("Weight loaded: %d", weight_path)
# names = [weight.name for layer in model.layers for weight in layer.weights]
# print(names)
with open('/content/drive/MyDrive/kernel_weight_of_dualfc.npy', 'rb') as f:
    kernel_weight = np.load(f)

original_kernel = get_original_dualfc_weight()
hosvd_error = tl.norm(original_kernel - kernel_weight)/tl.norm(kernel_weight)


core_name = 'dual_fc/hosvd_core:0'
factor_0_name = 'dual_fc/hosvd_factor_0:0'
factor_1_name = 'dual_fc/hosvd_factor_1:0'
factor_2_name = 'dual_fc/hosvd_factor_2:0'


# print(dir(model.get_layer("dual_fc")))
core_weight = get_weights_by_name(model, core_name).numpy()
factor_0_weight = get_weights_by_name(model, factor_0_name).numpy()
factor_1_weight = get_weights_by_name(model, factor_1_name).numpy()
factor_2_weight = get_weights_by_name(model, factor_2_name).numpy()


rec = tenalg.multi_mode_dot(core_weight, [factor_0_weight, factor_1_weight, factor_2_weight])
rec_error = tl.norm(rec - kernel_weight)/tl.norm(kernel_weight)

print("hosvd -- rec error:  ", hosvd_error)
print("after tuning -- rec error:  ", rec_error)



kernel_c_data = printVector(rec, "dual_fc_weights")
with open('kernel_data.c', 'w') as f:
    f.write(kernel_c_data)

with open('original_kernel_data.c', 'w') as f:
    f.write(printVector(kernel_weight, "dual_fc_weights"))
