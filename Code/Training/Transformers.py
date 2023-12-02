import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
from Preprocess import generate_training_sequences, SEQUENCE_LENGTH
#Similar to LSTMs you can change below parameters accordingly
OUTPUT_UNITS = 38
NUM_UNITS = 256  # This is the hidden size of the transformer model
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 64
SAVE_MODEL_PATH = "/Users/keerthanagolla/Desktop/RICE/SEM-1/StatisticalML/MusicGenerator/PreprocessedErnDataset/transformermodel.h5"
# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transformer_model = TFBertModel.from_pretrained('bert-base-uncased')
def build_model(output_units, num_units, loss, learning_rate):
    """Builds and compiles model
    :param output_units (int): Num output units
    :param num_units (int): Hidden size of the transformer model
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply
    :return model (tf model): Where the magic happens :D
    """
    print("inside buildmodel")
    # create the model architecture
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    print("bert embeddings")
    # BERT model embeddings
    embeddings = transformer_model(input_ids, attention_mask=attention_mask)[0]

    # Pooling layer (you can modify this based on your specific needs)
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(embeddings)
    output = tf.keras.layers.Dense(output_units, activation="softmax")(pooled_output)
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    # compile model
    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])
    model.summary()
    return model


def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    # generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    # build the network
    model = build_model(output_units, num_units, loss, learning_rate)
    progbar_logger = tf.keras.callbacks.ProgbarLogger()
    # create a dictionary with input names as keys and input data as values
    input_data = {"input_ids": inputs[:, :, 0], "attention_mask": inputs[:, :, 1]}
    # train the model using the dictionary
    print("model fit")
    model.fit(
        input_data,  # Pass the dictionary as input
        targets,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[progbar_logger]
    )
    # save the model
    model.save(SAVE_MODEL_PATH)

if __name__ == "__main__":
    print("start")
    train()
    print("done")
