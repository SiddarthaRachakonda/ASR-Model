from datasets import load_dataset
from datasets import Audio
import tensorflow as tf
from matplotlib import pyplot as plt


model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(
    f'{model_name}.zip',
    f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
    cache_dir='.', cache_subdir='', extract=True
)

def load_tokenizer():
    tokenizers = tf.saved_model.load(model_name)
    return tokenizers


def load_data():
    dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    COLUMNS_TO_KEEP = ["sentence", "audio"]
    all_columns = dataset.column_names
    columns_to_remove = set(all_columns) - set(COLUMNS_TO_KEEP)
    dataset = dataset.remove_columns(columns_to_remove)

    return dataset

# Plot the distribution of sentence lengths
def plot_sentence_lengths(dataset):
    tokenizer = load_tokenizer()
    sentence_lengths = []
    for example in dataset:
        sentence_lengths.append(len(tokenizer(example['sentence']).numpy()))

    # Print in csv
    with open('sentence_lengths.csv', 'w') as f:
        for length in sentence_lengths:
            f.write(f"{length}\n")

    plt.hist(sentence_lengths, bins=30)
    plt.show()



if __name__ == "__main__":
    data = load_data()
    print("Total tokens in the dataset:", plot_sentence_lengths(data))

