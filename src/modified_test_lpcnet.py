order = 16

pcm = np.zeros((nb_frames*pcm_chunk_size, ))
fexc = np.zeros((1, 1, 2), dtype='float32')
iexc = np.zeros((1, 1, 1), dtype='int16')
state1 = np.zeros((1, model.rnn_units1), dtype='float32')
state2 = np.zeros((1, model.rnn_units2), dtype='float32')

mem = 0
coef = 0.85

fout = open(out_file, 'wb')

mems = list()

skip = order + 1
for c in range(0, nb_frames):
    cfeat = enc.predict([features[c:c+1, :, :nb_used_features], periods[c:c+1, :, :]])
    for fr in range(0, feature_chunk_size):
        f = c*feature_chunk_size + fr
        a = features[c, fr, nb_features-order:]
        for i in range(skip, frame_size):
            pred = -sum(a*pcm[f*frame_size + i - 1:f*frame_size + i - order-1:-1])
            fexc[0, 0, 1] = lin2ulaw(pred)

            p, state1, state2 = dec.predict([fexc, iexc, cfeat[:, fr:fr+1, :], state1, state2])
            #Lower the temperature for voiced frames to reduce noisiness
            p *= np.power(p, np.maximum(0, 1.5*features[c, fr, 37] - .5))
            p = p/(1e-18 + np.sum(p))
            #Cut off the tail of the remaining distribution
            p = np.maximum(p-0.002, 0).astype('float64')
            p = p/(1e-8 + np.sum(p))

            iexc[0, 0, 0] = np.argmax(np.random.multinomial(1, p[0,0,:], 1))
            pcm[f*frame_size + i] = pred + ulaw2lin(iexc[0, 0, 0])
            fexc[0, 0, 0] = lin2ulaw(pcm[f*frame_size + i])
            mem = coef*mem + pcm[f*frame_size + i]
            
            #np.array([np.round(mem)], dtype='int16').tofile(fout)
            print("frame index c: ", c, "/", nb_frames, end="\t")
            print("feature index fr: ", fr, "/", feature_chunk_size, end="\t")
            print("i: ", i, "/", frame_size)

            print(mem)
        skip = 0

