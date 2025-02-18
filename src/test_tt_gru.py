from TuykiTTRNN import TT_GRU
from keras import initializers
import numpy as np
import tensorflow as tf

inp = tf.Variable(np.random.rand(11, 1, 16*32), dtype=float)
tt_input_shape=[16, 32]
tt_output_shape=[4, 4]

tt_ranks=[1, 8, 1]
rnn_layer = TT_GRU(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape, tt_ranks=tt_ranks, debug=True)

with tf.GradientTape() as tape:
  out = rnn_layer(inp)
dy_dx = tape.gradient(out, inp)
print(dy_dx)
print(dy_dx.numpy())

# print("output shape: ", out.shape)


print("------------------------------------------------------------------------------------------------")

# with open('/content/drive/MyDrive/recurrent_kernel_weight_of_grub.npy', 'rb') as f:
#         recurrent_kernel_weight = np.load(f)
  
# rnn2 = TT_GRU(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape, tt_ranks=tt_ranks, 
#                   return_sequences=True, return_state=True, 
#                   recurrent_regularizer=initializers.Constant(recurrent_kernel_weight),
#                   name='tt_gru_b') 

# tt_ranks2=[1, 8, 1]
# rnn_layer2 = TT_GRU(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape, tt_ranks=tt_ranks2, debug=True)
# rnn_layer2(inp)

#tt_input_shape=[16, 32], tt_output_shape=[8, 8], tt_ranks=[1, 3, 1],return_sequences=False, debug=True, dropout=.25, recurrent_dropout=.25, activation='tanh'
