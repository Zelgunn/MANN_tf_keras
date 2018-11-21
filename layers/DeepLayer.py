from keras.layers import Layer
import tensorflow as tf


class DeepLayer(Layer):
    def build_layer_weights(self, name, shape,
                            dtype=None,
                            kernel_initializer="glorot_uniform",
                            bias_initializer="zeros",
                            kernel_regularizer=None,
                            bias_regularizer=None,
                            kernel_constraint=None,
                            bias_constraint=None,
                            trainable=True,
                            use_bias=True):
        kernel = self.add_weight(name="{0}_kernel".format(name), shape=shape, dtype=dtype,
                                 initializer=kernel_initializer, regularizer=kernel_regularizer, trainable=trainable,
                                 constraint=kernel_constraint)
        if use_bias:
            bias = self.add_weight(name="{0}_bias".format(name), shape=shape[1:], dtype=dtype,
                                   initializer=bias_initializer, regularizer=bias_regularizer, trainable=trainable,
                                   constraint=bias_constraint)
        else:
            bias = None
        return LayerWeights(kernel, bias)


class LayerWeights(object):
    def __init__(self, kernel: tf.Variable, bias: tf.Variable):
        self.kernel = kernel
        self.bias = bias

    def __call__(self, inputs, **kwargs):
        output = inputs @ self.kernel
        if self.bias is not None:
            output = output + self.bias
        return output
