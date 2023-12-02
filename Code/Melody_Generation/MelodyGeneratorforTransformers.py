import json
import numpy as np
import keras
import music21 as m21
from Preprocess import SEQUENCE_LENGTH, MAPPING_PATH
import tensorflow as tf
import music21 as m21
from transformers import BertTokenizer, TFBertModel
class MelodyGenerator:
    """A class that wraps the LSTM model and offers utilities to generate melodies.
    Similar to the melody generator for LSTM , here also you can change the model path"""
#/Users/keerthanagolla/Desktop/RICE/SEM-1/StatisticalML/MusicGenerator/PreprocessedErnDataset/model.h5
    def __init__(self, model_path="/Users/keerthanagolla/Desktop/RICE/SEM-1/StatisticalML/MusicGenerator/transformermodel.h5"):
        """Constructor that initialises TensorFlow model"""
        tf.keras.utils.get_custom_objects()['TFBertModel'] = TFBertModel
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)
        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)
        self._start_symbols = ["/"] * SEQUENCE_LENGTH
    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        """Generates a melody using the DL model and returns a midi file.
        :param seed (str): Melody seed with the notation used to encode the dataset
        :param num_steps (int): Number of steps to be generated
        :param max_sequence_len (int): Max number of steps in seed to be considered for generation
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.
        :return melody (list of str): List with symbols representing a melody
        """
        # create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]
        for _ in range(num_steps):
            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]
            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # reshape the onehot_seed to match the expected input shape (None, max_sequence_length, num_classes)
            onehot_seed = onehot_seed.reshape(1, -1, len(self._mappings))
            # make a prediction
            probabilities = self.model.predict([onehot_seed[:, :, 0], onehot_seed[:, :, 1]])[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self._sample_with_temperature(probabilities, temperature)
            # update seed
            seed.append(output_int)
            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]
            # check whether we're at the end of a melody
            if output_symbol == "/":
                break
            # update melody
            melody.append(output_symbol)
        return melody

    def _sample_with_temperature(self, probabilites, temperature):
        """Samples an index from a probability array reapplying softmax using temperature
        :param predictions (nd.array): Array containing probabilities for each of the possible outputs.
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.
        :return index (int): Selected output symbol
        """
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))
        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)
        return index


    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="transformermel.midi"):
        """Converts a melody into a MIDI file
        :param melody (list of str):
        :param min_duration (float): Duration of each time step in quarter length
        :param file_name (str): Name of midi file
        :return:
        """
        # create a music21 stream
        stream = m21.stream.Stream()
        start_symbol = None
        step_counter = 1
        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):
            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):
                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1
                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                    stream.append(m21_event)
                    # reset the step counter
                    step_counter = 1
                start_symbol = symbol
            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1
        # write the m21 stream to a midi file
        stream.write(format, file_name)
if __name__ == "__main__":
    mg = MelodyGenerator()
    #Similar to LSTM melody generator you can specify seed accordingly.
    seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.3)
    print(melody)
    mg.save_melody(melody)
