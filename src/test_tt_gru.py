from TTLayer import *
from TuykiTTRNN import *

rnn_layer = TT_GRU(tt_input_shape=[16, 32], tt_output_shape=[8, 8], tt_ranks=[1, 3, 1],
                   return_sequences=False, debug=True,
                   dropout=.25, recurrent_dropout=.25, activation='tanh')
