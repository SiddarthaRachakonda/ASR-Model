import tensorflow as tf
import numpy as np


def positional_encoding(length, d_model, max_timescale=10000):
    """Positional Encoding using sines and cos"""
    log_timescale_increment = np.log(max_timescale) / (d_model // 2 - 1)
    inv_timescales = tf.exp(-log_timescale_increment * tf.range(d_model // 2, dtype=tf.float32))
    scaled_time = tf.range(length, dtype=tf.float32)[:, tf.newaxis] * inv_timescales[tf.newaxis, :]
    return tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)


class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
    self.pos_encoding = positional_encoding(length=2048, d_model=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x


class EncoderPositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.pos_encoding = positional_encoding(length=2048, d_model=d_model)

  def call(self, x):
    length = tf.shape(x)[1]
    # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x
