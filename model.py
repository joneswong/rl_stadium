from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf
from functools import partial


def get_model(inputs,
              num_outputs,
              options=None,
              state_in=None,
              seq_lens=None):
    """Returns a suitable model conforming to given input and output specs.

    Args:
        inputs (Tensor): The input tensor to the model.
        num_outputs (int): The size of the output vector of the model.
        options (dict): Optional args to pass to the model constructor.
        state_in (list): Optional RNN state in tensors.
        seq_in (Tensor): Optional RNN sequence length tensor.

    Returns:
        model (Model): Neural network model.
    """

    options = options or {}
    model = handle_get_model(inputs, num_outputs, options, state_in,
                                    seq_lens)

    if options.get("use_lstm"):
        model = LSTM(model.last_layer, num_outputs, options, state_in,
                     seq_lens)

    return model


def handle_get_model(inputs, num_outputs, options, state_in, seq_lens):
    #if "custom_model" in options:
    #    model = options["custom_model"]
    #    print("Using custom model {}".format(model))
    #    return _global_registry.get(RLLIB_MODEL, model)(
    #        inputs,
    #        num_outputs,
    #        options,
    #        state_in=state_in,
    #        seq_lens=seq_lens)

    obs_rank = len(inputs.shape) - 1

    # num_outputs > 1 used to avoid hitting this with the value function
    #if isinstance(
    #        options.get("custom_options", {}).get(
    #            "multiagent_fcnet_hiddens", 1), list) and num_outputs > 1:
    #    return MultiAgentFullyConnectedNetwork(inputs, num_outputs,
    #                                           options)

    if obs_rank > 1:
        return VisionNetwork(inputs, num_outputs, options)

    return FullyConnectedNetwork(inputs, num_outputs, options)


class Model(object):
    """Defines an abstract network model for use with RLlib.

    Models convert input tensors to a number of output features. These features
    can then be interpreted by ActionDistribution classes to determine
    e.g. agent action values.

    The last layer of the network can also be retrieved if the algorithm
    needs to further post-processing (e.g. Actor and Critic networks in A3C).

    Attributes:
        inputs (Tensor): The input placeholder for this model, of shape
            [BATCH_SIZE, ...].
        outputs (Tensor): The output vector of this model, of shape
            [BATCH_SIZE, num_outputs].
        last_layer (Tensor): The network layer right before the model output,
            of shape [BATCH_SIZE, N].
        state_init (list): List of initial recurrent state tensors (if any).
        state_in (list): List of input recurrent state tensors (if any).
        state_out (list): List of output recurrent state tensors (if any).
        seq_lens (Tensor): The tensor input for RNN sequence lengths. This
            defaults to a Tensor of [1] * len(batch) in the non-RNN case.

    If `options["free_log_std"]` is True, the last half of the
    output layer will be free variables that are not dependent on
    inputs. This is often used if the output of the network is used
    to parametrize a probability distribution. In this case, the
    first half of the parameters can be interpreted as a location
    parameter (like a mean) and the second half can be interpreted as
    a scale parameter (like a standard deviation).
    """

    def __init__(self,
                 inputs,
                 num_outputs,
                 options,
                 state_in=None,
                 seq_lens=None):
        self.inputs = inputs

        # Default attribute values for the non-RNN case
        self.state_init = []
        self.state_in = state_in or []
        self.state_out = []
        if seq_lens is not None:
            self.seq_lens = seq_lens
        else:
            self.seq_lens = tf.placeholder(
                dtype=tf.int32, shape=[None], name="seq_lens")

        if options.get("free_log_std", False):
            assert num_outputs % 2 == 0
            num_outputs = num_outputs // 2
        self.outputs, self.last_layer = self._build_layers(
            inputs, num_outputs, options)
        if options.get("free_log_std", False):
            log_std = tf.get_variable(
                name="log_std",
                shape=[num_outputs],
                initializer=tf.zeros_initializer)
            self.outputs = tf.concat(
                [self.outputs, 0.0 * self.outputs + log_std], 1)

    def _build_layers(self):
        """Builds and returns the output and last layer of the network."""
        raise NotImplementedError


class FullyConnectedNetwork(Model):
    """Generic fully connected network."""

    def _build_layers(self, inputs, num_outputs, options):
        hiddens = options.get("fcnet_hiddens", [256, 256])
        activation = get_activation_fn(options.get("fcnet_activation", "tanh"))

        with tf.name_scope("fc_net"):
            i = 1
            last_layer = inputs
            for size in hiddens:
                label = "fc{}".format(i)
                last_layer = slim.fully_connected(
                    last_layer,
                    size,
                    weights_initializer=normc_initializer(1.0),
                    activation_fn=activation,
                    scope=label)
                i += 1
            label = "fc_out"
            output = slim.fully_connected(
                last_layer,
                num_outputs,
                weights_initializer=normc_initializer(0.01),
                activation_fn=None,
                scope=label)
            return output, last_layer


class VisionNetwork(Model):
    """Generic vision network."""

    def _build_layers(self, inputs, num_outputs, options):
        filters = options.get("conv_filters")
        if not filters:
            filters = get_filter_config(options)

        activation = get_activation_fn(options.get("conv_activation", "relu"))

        with tf.name_scope("vision_net"):
            for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
                inputs = slim.conv2d(
                    inputs,
                    out_size,
                    kernel,
                    stride,
                    activation_fn=activation,
                    scope="conv{}".format(i))
            out_size, kernel, stride = filters[-1]
            fc1 = slim.conv2d(
                inputs,
                out_size,
                kernel,
                stride,
                activation_fn=activation,
                padding="VALID",
                scope="fc1")
            fc2 = slim.conv2d(
                fc1,
                num_outputs, [1, 1],
                activation_fn=None,
                normalizer_fn=None,
                scope="fc2")
            return flatten(fc2), flatten(fc1)


def get_filter_config(options):
    filters_84x84 = [
        [16, [8, 8], 4],
        [32, [4, 4], 2],
        [256, [11, 11], 1],
    ]
    filters_42x42 = [
        [16, [4, 4], 2],
        [32, [4, 4], 2],
        [256, [11, 11], 1],
    ]
    dim = options.get("dim", 84)
    if dim == 84:
        return filters_84x84
    elif dim == 42:
        return filters_42x42
    else:
        raise ValueError(
            "No default configuration for image size={}".format(dim) +
            ", you must specify `conv_filters` manually as a model option.")


def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def get_activation_fn(name):
    return getattr(tf.nn, name)


def conv2d(x,
           num_filters,
           name,
           filter_size=(3, 3),
           stride=(1, 1),
           pad="SAME",
           dtype=tf.float32,
           collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [
            filter_size[0], filter_size[1],
            int(x.get_shape()[3]), num_filters
        ]

        # There are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit.
        fan_in = np.prod(filter_shape[:3])
        # Each unit in the lower layer receives a gradient from: "num output
        # feature maps * filter height * filter width" / pooling size.
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # Initialize weights with random weights.
        w_bound = np.sqrt(6 / (fan_in + fan_out))

        w = tf.get_variable(
            "W",
            filter_shape,
            dtype,
            tf.random_uniform_initializer(-w_bound, w_bound),
            collections=collections)
        b = tf.get_variable(
            "b", [1, 1, 1, num_filters],
            initializer=tf.constant_initializer(0.0),
            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(
        name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(
        name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
