import tensorflow as tf
from keras.layers import RNN
import keras.backend as K

from layers.ExternalMemoryCell import ExternalMemoryCell


class MemoryAugmentedLayer(RNN):
    def __init__(self,
                 memory_width,
                 memory_height,
                 usage_decay,
                 read_heads_count,
                 controller_units,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        cell = ExternalMemoryCell(memory_width=memory_width,
                                  memory_height=memory_height,
                                  usage_decay=usage_decay,
                                  read_heads_count=read_heads_count,
                                  controller_units=controller_units,
                                  **kwargs)

        super(MemoryAugmentedLayer, self).__init__(cell,
                                                   return_sequences=return_sequences,
                                                   return_state=return_state,
                                                   go_backwards=go_backwards,
                                                   stateful=stateful,
                                                   unroll=unroll,
                                                   **kwargs)

    def get_initial_state(self, inputs: tf.Tensor):
        # memory, cell_state, hidden_state, read_weights, usage_weights, read_vector
        zeros = K.zeros_like(inputs)
        zeros = K.sum(zeros, axis=(1, 2))
        zeros = K.expand_dims(zeros)
        zeros_dims = self.cell.state_size[1:]
        zeros = [K.tile(zeros, [1, dim]) for dim in zeros_dims]

        memory = K.ones_like(inputs) * tf.constant(1e-6)
        memory = K.mean(memory, axis=(1, 2))
        memory = K.expand_dims(memory)
        memory = K.tile(memory, [1, self.cell.state_size[0]])
        return [memory] + zeros
