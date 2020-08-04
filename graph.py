
import tensorflow as tf
from tensorflow.keras import activations, initializers, constraints
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K


class GraphConvolution(tf.keras.layers.Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, units, support=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.supports_masking = True

        self.support = support
        assert support >= 1

    def compute_output_shape(self, input_shapes):
        #features_shape = input_shapes[0]
        #output_shape = (features_shape[0], self.units)
        return input_shapes[0][:2] + (self.units,)
        #return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        #features_shape = input_shapes[0]
        #assert len(features_shape) == 2
        input_dim = input_shapes[0].as_list()[2]
        #input_dim = features_shape[1]

        self.kernel = self.add_weight(shape=(input_dim * self.support,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        features = inputs[0]
        basis = inputs[1:]
        #basis = K.cast(basis, K.floatx())
        #features, basis = inputs
        
        supports = list()
        for i in range(self.support):
            supports.append(K.batch_dot(basis[i], features, axes=[2,1]))
        supports = K.concatenate(supports, axis=2)
        output = K.dot(supports, self.kernel)

        if self.use_bias:
            output += self.bias
        return self.activation(output)

    def get_config(self):
        config = {'units': self.units,
                  'support': self.support,
                  'activation': tf.keras.activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': tf.keras.initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': tf.keras.initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': tf.keras.regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': tf.keras.regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': tf.keras.regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': tf.keras.constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint)
        }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
