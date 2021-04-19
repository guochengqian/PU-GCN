import tensorflow as tf


def instance_norm(net, weight_decay=0.00001):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keepdims=True)

    shift = tf.get_variable('shift', shape=var_shape,
                            initializer=tf.zeros_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    scale = tf.get_variable('scale', shape=var_shape,
                            initializer=tf.ones_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    epsilon = 1e-3
    normalized = (net - mu) / tf.square(sigma_sq + epsilon)
    return scale * normalized + shift


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           use_bn=False,
           use_ibn=False,
           bn_decay=None,
           use_bias=True,
           is_training=None,
           reuse=None):
    """ 1D convolution with non-linear operation.

    Args:
        inputs: 3-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: int
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope, reuse=reuse):
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.conv1d(inputs, num_output_channels, kernel_size, stride, padding,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   use_bias=use_bias, reuse=None)
        assert not (use_bn and use_ibn)
        if use_bn:
            outputs = tf.layers.batch_normalization(
                outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)
            # outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
        if use_ibn:
            outputs = instance_norm(outputs)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def conv2d(inputs,
           num_output_channels,
           kernel_size=[1, 1],
           scope=None,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           use_bn=False,
           use_ibn=False,
           bn_decay=None,
           use_bias=True,
           is_training=None,
           reuse=tf.AUTO_REUSE):
    """ 2D convolution with non-linear operation.

    Args:
      inputs: 4-D tensor variable BxHxWxC
      num_output_channels: int
      kernel_size: a list of 2 ints
      scope: string
      stride: a list of 2 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      use_bn: bool, whether to use batch norm
      use_ibn: bool, whether to use instance norm
      use_bias: bool, whether to add bias
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope, reuse=reuse) as sc:
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.conv2d(inputs, num_output_channels, kernel_size, stride, padding,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                   use_bias=use_bias)
        assert not (use_bn and use_ibn)
        if use_bn:
            outputs = tf.layers.batch_normalization(outputs, momentum=bn_decay, training=is_training, renorm=False,
                                                    fused=True)
        if use_ibn:
            outputs = instance_norm(outputs)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.00001,
                    activation_fn=tf.nn.relu,
                    use_bn=False,
                    bn_decay=None,
                    use_bias=True,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """

    with tf.variable_scope(scope) as sc:
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.dense(inputs, num_outputs,
                                  use_bias=use_bias, kernel_initializer=initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  reuse=None)

        if use_bn:
            outputs = tf.layers.batch_normalization(outputs, momentum=bn_decay, training=is_training, renorm=False)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs
