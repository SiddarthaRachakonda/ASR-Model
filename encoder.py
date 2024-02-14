import tensorflow as tf
from attention import GlobalSelfAttention
from feedforward import FeedForward
from positional_encoding import PositionalEmbedding

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x


def conv_block(filters = 64, kernel_size=3, activation = 'gelu', padding = 'same', input_shape = (None, 128)):
    model = tf.keras.models.Sequential(
                tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding, input_shape=input_shape),
                tf.keras.layers.Conv1D(filters=filters * 2, kernel_size=kernel_size, activation=activation, padding=padding),
                tf.keras.layers.Dense(128)) # Modify to the required output shape
    return model

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.conv_layer = conv_block()

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size, d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):

    # `x` is token-IDs shape: (batch, seq_len)
    x = self.conv_layer(x)

    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.