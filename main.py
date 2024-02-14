
from transformer import Transformer


def main():
    # Model parameters
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    # Input and target vocabulary size
    input_vocab_size = 8500
    target_vocab_size = 8000

    # Create a transformer model
    transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                              input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size,
                              dropout_rate=dropout_rate)

    # Compile the model
    transformer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    transformer.summary()