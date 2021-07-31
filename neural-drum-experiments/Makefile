black:
	black *.py

clean_rnn:
	rm -rf model/trnnsient.h5
	rm -rf logdir/trnnsient.ckpt

retrain_rnn: clean_rnn
	./rnn_model.py ./training_bfcc.hdf5

clean_wavenet:
	rm -rf model/wavenet.h5
	rm -rf logdir/wavenet.h5

retrain_wavenet: clean_wavenet
	./wavenet_model.py ./training_waveform.hdf5

fmt: black

.PHONY: black fmt clean_rnn retrain_rnn clean_wavenet retrain_rnn
