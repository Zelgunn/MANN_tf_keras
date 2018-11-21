import tensorflow as tf


def interpolate(a: tf.Tensor, b: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    return b * t + a * (tf.constant(1.0) - t)


def top_k_mask(input_tensor: tf.Tensor, k=1):
    batch_size = tf.shape(input_tensor)[0]

    _, indices = tf.nn.top_k(input_tensor, k=k)
    batch_indices = tf.meshgrid(tf.range(k), tf.range(batch_size))[1]
    indices = tf.stack([batch_indices, indices], axis=2)
    indices = tf.reshape(indices, [batch_size * k, 2])
    indices = tf.cast(indices, tf.int64)

    values = tf.ones(shape=[batch_size * k], dtype=tf.int64)
    dense_shape = tf.shape(input_tensor, out_type=tf.int64)

    sparse_representation = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    sparse_representation = tf.sparse_reorder(sparse_representation)
    return tf.sparse.to_dense(sparse_representation)


def bot_k_mask(input_tensor: tf.Tensor, k=1, inverse=False):
    mask = top_k_mask(- input_tensor, k)
    if inverse:
        mask = tf.ones_like(input_tensor, dtype=tf.int64) - mask
    return mask
