from keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.python.keras import activations, initializers, regularizers, constraints
import numpy as np
import tensorflow as tf
import math

class MDense(Layer):
    def __init__(self, outputs,
                 channels=2,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MDense, self).__init__(**kwargs)
        self.units = outputs
        self.channels = channels
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        with open('/content/drive/MyDrive/core_kernel_weight_of_dualfc.npy', 'rb') as f:
            core_weight = np.load(f)

        with open('/content/drive/MyDrive/factor_0_kernel_weight_of_dualfc.npy', 'rb') as f:
            factor_0_weight = np.load(f)

        with open('/content/drive/MyDrive/factor_1_kernel_weight_of_dualfc.npy', 'rb') as f:
            factor_1_weight = np.load(f)

        with open('/content/drive/MyDrive/factor_2_kernel_weight_of_dualfc.npy', 'rb') as f:
            factor_2_weight = np.load(f)
        
        self.core     = tf.Variable(initializer=initializers.Constant(core_weight), name='hosvd_core', trainable=True)
        self.factor_0 = tf.Variable(initializer=initializers.Constant(factor_0_weight), name='hosvd_factor_0', trainable=True)
        self.factor_1 = tf.Variable(initializer=initializers.Constant(factor_1_weight), name='hosvd_factor_1', trainable=True)
        self.factor_2 = tf.Variable(initializer=initializers.Constant(factor_2_weight), name='hosvd_factor_2', trainable=True)

        self.kernel = self.add_weight(shape=(self.units, input_dim, self.channels), # Expected (pcm_levels ~ 2**8=256, 54, 2)
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units, self.channels), # Expected (pcm_levels ~ 2**8=256, 2)
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=False)
        else:
            self.bias = None
        self.factor = self.add_weight(shape=(self.units, self.channels), # Expected (pcm_levels ~ 2**8=256, 2)
                                    initializer='ones',
                                    name='factor',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint, 
                                    trainable=False)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True


    def call(self, inputs):
        r0 = tf.einsum('ijn,ki->kjn', self.core, self.factor_0)
        r01 = tf.einsum('ijn,kj->ikn', r0, self.factor_1)
        self.kernel = tf.einsum('ijn,kn->ijk', r01, self.factor_2)

        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = output + self.bias
        output = K.tanh(output) * self.factor
        output = K.sum(output, axis=-1)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
