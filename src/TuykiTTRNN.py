__author__ = "Yinchong Yang"
__copyright__ = "Siemens AG, 2017"
__licencse__ = "MIT"
__version__ = "0.1"

"""
MIT License
Copyright (c) 2017 Siemens AG
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from tensorflow.python.keras.layers.recurrent import SimpleRNN
from keras import backend as K
from tensorflow.keras.layers import InputSpec

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints

class TT_GRU(SimpleRNN):
    """
    # Arguments
        tt_input_shape: a list of shapes, the product of which should be equal to the input dimension
        tt_output_shape: a list of shapes of the same length as tt_input_shape,
            the product of which should be equal to the output dimension
        tt_ranks: a list of length len(tt_input_shape)+1, the first and last rank should only be 1
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    # References
        - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        - [Tensor Train Recurrent Neural Networks for Video Classification](https://arxiv.org/abs/1707.01786)
    """
    def __init__(self,
                 units,
                 tt_input_shape, tt_output_shape, tt_ranks,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 debug=False,
                 init_seed=11111986,
                 **kwargs):
        super(TT_GRU, self).__init__(units = units, **kwargs)

        self._units = units
        self._activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))

        self.state_spec = InputSpec(shape=(None, self._units))
        self.debug = debug
        self.init_seed = init_seed

        tt_input_shape = np.array(tt_input_shape)
        tt_output_shape = np.array(tt_output_shape)
        tt_ranks = np.array(tt_ranks)
        self.num_dim = tt_input_shape.shape[0]
        self.tt_input_shape = tt_input_shape
        self.tt_output_shape = tt_output_shape
        self.tt_ranks = tt_ranks
        self.debug = debug

    def build(self, input_shape):

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec[0] = InputSpec(shape=(batch_size, None, self.input_dim))

        self.states = [None]
        if self.stateful:
            self.reset_states()

        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.tt_output_shape[0] *= 3

        num_inputs = int(np.prod(input_shape[2::])) # instead of [1::]
        if np.prod(self.tt_input_shape) != num_inputs:
            raise ValueError("The size of the input tensor (i.e. product "
                             "of the elements in tt_input_shape) should "
                             "equal to the number of input neurons %d." %
                             (num_inputs))
        if self.tt_input_shape.shape[0] != self.tt_output_shape.shape[0]:
            raise ValueError("The number of input and output dimensions "
                             "should be the same.")
        if self.tt_ranks.shape[0] != self.tt_output_shape.shape[0] + 1:
            raise ValueError("The number of the TT-ranks should be "
                             "1 + the number of the dimensions.")

        if self.debug:
            print ('tt_input_shape = ', str( self.tt_input_shape ))
            print ('tt_output_shape = ', str( self.tt_output_shape ))
            print ('tt_ranks = ', str( self.tt_ranks ))

        np.random.seed(self.init_seed)
        total_length = np.sum(self.tt_input_shape * self.tt_output_shape *
                               self.tt_ranks[1:] * self.tt_ranks[:-1])
        local_cores_arr = np.random.randn(total_length)
        self.kernel = self.add_weight((total_length, ),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight((np.prod(self.tt_output_shape), ),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.inds = np.zeros(self.num_dim).astype('int32')
        self.shapes = np.zeros((self.num_dim, 2)).astype('int32')
        self.cores = [None] * self.num_dim

        for k in range(self.num_dim - 1, -1, -1):
            self.shapes[k] = (self.tt_input_shape[k] * self.tt_ranks[k + 1],
                              self.tt_ranks[k] * self.tt_output_shape[k])
            self.cores[k] = self.kernel[self.inds[k]:self.inds[k] + np.prod(self.shapes[k])]
            if 0 < k:
                self.inds[k - 1] = self.inds[k] + np.prod(self.shapes[k])
        if self.debug:
            print ('self.shapes = ', str(self.shapes))

        self.TT_size = total_length
        self.full_size = (np.prod(self.tt_input_shape) * np.prod(self.tt_output_shape))
        self.compress_factor = 1. * self.TT_size / self.full_size
        print ('Compression factor = ' , str(self.TT_size) , ' / ' \
              , str(self.full_size) , ' = ' , str(self.compress_factor))

        self.recurrent_kernel = self.add_weight(
            shape=(self._units, self._units*3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.inds = np.zeros(self.num_dim).astype('int32')
        self.shapes = np.zeros((self.num_dim, 2)).astype('int32')
        self.cores = [None]*(self.num_dim)

        for k in range(self.num_dim -1, -1, -1):
            self.shapes[k] = (self.tt_input_shape[k] * self.tt_ranks[k + 1],
                              self.tt_ranks[k] * self.tt_output_shape[k])
            self.cores[k] = self.kernel[self.inds[k]:self.inds[k]+np.prod(self.shapes[k])]
            if 0 < k:
                self.inds[k-1] = self.inds[k] + np.prod(self.shapes[k])

        self.compress_factor = 1.*(local_cores_arr.size) / \
                               (np.prod(self.tt_input_shape)*np.prod(self.tt_output_shape))

        print ('Compressrion factor = ', str(self.compress_factor))

        self.built = True

    def preprocess_input(self, x, training=None):
        return x

    def get_constants(self, inputs, training=None):
        constants = []
        constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0. < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self._units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)
            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(3)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        dp_mask = states[1]  # dropout matrices for recurrent units
        rec_dp_mask = states[2]

        res = x * dp_mask[0]
        for k in range(self.num_dim - 1, -1, -1):
            res = K.dot(K.reshape(res, (-1, self.shapes[k][0])),
                        K.reshape(self.cores[k], self.shapes[k])
                        )
            res = K.transpose(K.reshape(res, (-1, self.tt_output_shape[k])))
        res = K.transpose(K.reshape(res, (-1, K.shape(x)[0])))

        matrix_x = res

        if self.use_bias:
            matrix_x = K.bias_add(matrix_x, self.bias)
        matrix_inner = K.dot(h_tm1 * rec_dp_mask[0],
                             self.recurrent_kernel[:, :2 * self._units])
        x_z = matrix_x[:, :self._units]
        x_r = matrix_x[:, self._units: 2 * self._units]
        recurrent_z = matrix_inner[:, :self._units]
        recurrent_r = matrix_inner[:, self._units: 2 * self._units]

        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)

        x_h = matrix_x[:, 2 * self._units:]
        recurrent_h = K.dot(r * h_tm1 * rec_dp_mask[0],
                            self.recurrent_kernel[:, 2 * self._units:])
        hh = self._activation(x_h + recurrent_h)

        h = z * h_tm1 + (1 - z) * hh
        if 0. < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h]

    def get_config(self):
        config = {'units': self._units,
                  'activation': activations.serialize(self._activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(TT_GRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
