# MelodyGenerator_LSTM_BERT
Melody Generation using LSTM, transformer (using pre-trained BERT)

Step -1: Download the dataset (either from kaggle or hummdrum.org of kern/midi format

Step -2: Take the preprocess py file, change the path parameters accordingly, and run the code to generate one preprocessed file.

Step -3: Now using this single preprocessed file let's generate the models - Take the LSTM train py file, change the required path parameters along with that change the no.of input nodes according to the no.of mapping generated with your particular dataset, and run the file, it generates the LSTM model in h5 file.

Step -4: Similar to LSTM, Take the Transformers py file and Generate the Transformer hs file.

Step -5: Now let's generate the melody using the LSTM h5 path. Take the melody generator for LSTM and change the path variable to LSTM h5 path and it generates the midi file.

Step -6: Similar to LSTM, Take the melody generator for the Transformer and give the Transformer h5 path to generate the melody and save it in midi format.
