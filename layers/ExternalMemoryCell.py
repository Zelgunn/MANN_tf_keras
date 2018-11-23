import tensorflow as tf
import numpy as np
from typing import Tuple

from layers.DeepLayer import DeepLayer
from layers.utils import interpolate, bot_k_mask


class ExternalMemoryCell(DeepLayer):
    def __init__(self,
                 memory_width,
                 memory_height,
                 usage_decay,
                 read_heads_count,
                 controller_units,
                 **kwargs):
        super(ExternalMemoryCell, self).__init__(**kwargs)

        self.memory_width = memory_width
        self.memory_height = memory_height
        self.usage_decay = usage_decay
        self._usage_decay = tf.constant(usage_decay, dtype=tf.float32)
        self.read_heads_count = read_heads_count
        self.controller_units = controller_units

        self.input_to_hidden_kernel = None
        self.hidden_through_time_kernel = None
        self.prev_read_to_hidden_kernel = None
        self.hidden_state_bias = None
        self.key_layer = None
        self.add_layer = None
        self.write_interpolation_scalar_layer = None
        self.state_size = (self.memory_height * self.memory_width,
                           self.controller_units,
                           self.controller_units,
                           self.read_heads_count * self.memory_height,
                           self.memory_height,
                           self.read_heads_count * self.memory_width)
        self.output_size = self.controller_units + self.read_heads_count * self.memory_width

    def build(self, input_shape):
        input_dimension = input_shape[-1]

        self.input_to_hidden_kernel = self.add_weight(name="input_to_hidden_kernel",
                                                      shape=[input_dimension, self.controller_units * 4],
                                                      initializer="glorot_uniform")

        self.hidden_through_time_kernel = self.add_weight(name="hidden_through_time_kernel",
                                                          shape=[self.controller_units, self.controller_units * 4],
                                                          initializer="glorot_uniform")

        self.prev_read_to_hidden_kernel = self.add_weight(name="prev_read_to_hidden_kernel",
                                                          shape=[self.read_heads_count * self.memory_width,
                                                                 self.controller_units * 4],
                                                          initializer="glorot_uniform")

        self.hidden_state_bias = self.add_weight(name="hidden_state_bias", shape=[self.controller_units * 4],
                                                 initializer="zeros")

        self.key_layer = self.build_layer_weights(name="key", shape=[self.controller_units,
                                                                     self.read_heads_count * self.memory_width])

        self.add_layer = self.build_layer_weights(name="add", shape=[self.controller_units,
                                                                     self.read_heads_count * self.memory_width])

        self.write_interpolation_scalar_layer = self.build_layer_weights(name="write_interpolation_scalar",
                                                                         shape=[input_dimension, 1])

    def call(self, inputs, states):
        prev_memory, prev_cell_state, prev_hidden_state, prev_read_weights, prev_usage_weights, prev_read_vector = \
            states

        prev_memory = tf.reshape(prev_memory, [-1, self.memory_height, self.memory_width])
        prev_read_weights = tf.reshape(prev_read_weights, [-1, self.read_heads_count, self.memory_height])

        hidden_state, cell_state = self.controller_lstm_step(inputs, prev_hidden_state,
                                                             prev_cell_state, prev_read_vector)

        def query(layer):
            layer = layer(hidden_state)
            layer = tf.nn.tanh(layer)
            layer = tf.reshape(layer, shape=[-1, self.read_heads_count, self.memory_width])
            return layer

        key = query(self.key_layer)
        added = query(self.add_layer)

        similarity = ExternalMemoryCell.cosine_similarity(key, prev_memory)
        read_weights = tf.nn.softmax(similarity, axis=-1)

        read_vector = read_weights @ prev_memory
        read_vector = tf.reshape(read_vector, [-1, self.read_heads_count * self.memory_width])

        write_interpolation_scalar = tf.sigmoid(self.write_interpolation_scalar_layer(inputs))
        prev_read_usage_weights = tf.reduce_mean(prev_read_weights, axis=1)
        prev_least_used_weights = self.get_least_used_weights(prev_usage_weights)
        write_weights = interpolate(prev_least_used_weights, prev_read_usage_weights, write_interpolation_scalar)

        read_usage_weights = tf.reduce_mean(read_weights, axis=1)
        usage_weights = self._usage_decay * prev_usage_weights + read_usage_weights + write_weights
        least_used_weights = self.get_least_used_weights(usage_weights, inverse=True)
        least_used_weights_mul = tf.reshape(least_used_weights, [-1, self.memory_height, 1])

        write_weights = tf.expand_dims(write_weights, axis=2)
        write_key = tf.reduce_sum(added, axis=1, keepdims=True)
        memory = prev_memory * least_used_weights_mul + write_weights @ write_key

        output = tf.concat([hidden_state, read_vector], axis=-1)

        memory = tf.reshape(memory, [-1, self.memory_height * self.memory_width])
        read_weights = tf.reshape(read_weights, [-1, self.read_heads_count * self.memory_height])

        return output, [memory, cell_state, hidden_state, read_weights, usage_weights, read_vector]

    def controller_lstm_step(self,
                             inputs,
                             prev_hidden_state,
                             prev_cell_state,
                             prev_read_vector) -> Tuple[tf.Tensor, tf.Tensor]:
        pre_activations = inputs @ self.input_to_hidden_kernel
        pre_activations += prev_hidden_state @ self.hidden_through_time_kernel
        pre_activations += prev_read_vector @ self.prev_read_to_hidden_kernel
        pre_activations += self.hidden_state_bias

        forget_gate, input_gate, output_gate, update = tf.split(pre_activations, num_or_size_splits=4, axis=-1)

        forget_gate = tf.sigmoid(forget_gate)
        input_gate = tf.sigmoid(input_gate)
        output_gate = tf.sigmoid(output_gate)
        update = tf.tanh(update)

        cell_state = forget_gate * prev_cell_state + input_gate * update
        hidden_state = output_gate * tf.tanh(cell_state)

        return hidden_state, cell_state

    def get_least_used_weights(self, usage_weights, inverse=False):
        least_used_weights = bot_k_mask(usage_weights, k=self.read_heads_count, inverse=inverse)
        least_used_weights = tf.cast(least_used_weights, tf.float32)
        return least_used_weights

    def get_least_used_weights_alt(self, usage_weights):
        _, indices = tf.nn.top_k(usage_weights, k=self.read_heads_count)

    def get_config(self):
        config = {
            'memory_width': self.memory_width,
            'memory_height': self.memory_height,
            'usage_decay': self.usage_decay,
            'read_heads_count': self.read_heads_count,
            'controller_units': self.controller_units
        }
        base_config = super(ExternalMemoryCell, self).get_config()
        return {**config, **base_config}

    @staticmethod
    def cosine_similarity(key: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
        key = tf.nn.l2_normalize(key, axis=1)
        memory = tf.nn.l2_normalize(memory, axis=1)
        memory = tf.transpose(memory, [0, 2, 1])
        return key @ memory

    def initial_state(self, batch_size):
        memory = tf.constant(value=1e-4, shape=[batch_size, self.memory_height * self.memory_width],
                             name="Memory_initial")
        cell_state = tf.zeros(shape=[batch_size, self.controller_units], name="Controller_cell_state")
        hidden_state = tf.zeros(shape=[batch_size, self.controller_units], name="Controller_hidden_state")

        weight_vector = np.zeros([batch_size, self.memory_height])
        weight_vector[..., 0] = 1
        usage_weights = tf.constant(weight_vector, dtype=tf.float32, name="usage_weights")

        weight_vector = np.tile(weight_vector, self.read_heads_count)
        weight_vector = np.reshape(weight_vector, [batch_size, self.read_heads_count * self.memory_height])
        read_weights = tf.constant(weight_vector, dtype=tf.float32, name="read_weights")

        read_vector = tf.zeros(shape=[batch_size, self.read_heads_count * self.memory_width], name="read_vector")

        return memory, cell_state, hidden_state, read_weights, usage_weights, read_vector
