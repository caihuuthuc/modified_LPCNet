from TuykiTTRNN import TT_GRU

import numpy as np

tt_input_shape=[16, 32]

tt_output_shape=[8, 8]
tt_ranks=[1, 3, 1]
units = np.prod(np.array(tt_output_shape))

rnn_layer = TT_GRU(tt_input_shape=[16, 32], tt_output_shape=[8, 8], tt_ranks=[1, 3, 1], debug=True)

inp = np.random.rand(1, 16,32)
rnn_layer(inp)



#tt_input_shape=[16, 32], tt_output_shape=[8, 8], tt_ranks=[1, 3, 1],return_sequences=False, debug=True, dropout=.25, recurrent_dropout=.25, activation='tanh'
