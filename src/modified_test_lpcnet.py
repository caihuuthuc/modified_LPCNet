import lpcnet
import sys
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from ulaw import ulaw2lin, lin2ulaw
import keras.backend as K
import h5py

import tensorflow as tf
from tensorflow.python.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.compat.v1.Session(config=config))

model, enc, dec = lpcnet.new_lpcnet_model()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
#model.summary()

feature_file = sys.argv[1]
out_file = sys.argv[2]
frame_size = 160
nb_features = 55
nb_used_features = model.nb_used_features

features = np.fromfile(feature_file, dtype='float32')
features = features.reshape((-1, nb_features))

for i in features.shape[0]:
    feature = features[i,:]
    print(feature.shape)
    break
