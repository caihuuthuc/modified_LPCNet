from TuykiTTRNN import TT_GRU

import numpy as np

inp = np.random.rand(1, 1, 16*32)
tt_input_shape=[16, 32]
tt_output_shape=[4, 4]

tt_ranks=[1, 4, 1]
rnn_layer = TT_GRU(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape, tt_ranks=tt_ranks, debug=True)
rnn_layer(inp)

tt_ranks2=[1, 8, 1]
rnn_layer2 = TT_GRU(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape, tt_ranks=tt_ranks2, debug=True)
rnn_layer2(inp)

#tt_input_shape=[16, 32], tt_output_shape=[8, 8], tt_ranks=[1, 3, 1],return_sequences=False, debug=True, dropout=.25, recurrent_dropout=.25, activation='tanh'
