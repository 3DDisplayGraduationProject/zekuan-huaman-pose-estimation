"""
Defines networks.

@Encoder_resnet
@Encoder_resnet_v1_101
@Encoder_fc3_dropout

@Discriminator_separable_rotations

Helper:
@get_encoder_fn_separate
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.layers.python.layers.initializers import variance_scaling_initializer


def Encoder_resnet(x, is_training=True, weight_decay=0.001, reuse=False):
    """
    Resnet v2-50
    Assumes input is [batch, height_in, width_in, channels]!!
    Input:
    - x: N x H x W x 3
    - weight_decay: float
    - reuse: bool->True if test

    Outputs:
    - cam: N x 3
    - Pose vector: N x 72
    - Shape vector: N x 10
    - variables: tf variables
    """
    # from tensorflow.contrib.slim.python.slim.nets import resnet_v2
    from src import resnet_v2
    with slim.arg_scope(
            resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
        # net = slim.conv2d(x, 3, kernel_size=7, stride=2)
        # net = slim.batch_norm(net)
        # net = tf.nn.relu(net)
        # net = slim.max_pool2d(net, kernel_size=3, stride=2, padding="SAME")
        net = x
        # bottom-up
        net_c2, end_points = resnet_v2.resnet_v2(
            net,
            blocks = [resnet_v2.resnet_v2_block('block1', base_depth=64, num_units=3, stride=1)],
            # num_classes = 64,
            global_pool=False,
            is_training = is_training,
            reuse=False,
            scope='resnet_v2_101',
            post_norm_scope='postnorm_b1')
        net_c3, end_points = resnet_v2.resnet_v2(
            net_c2,
            blocks=[resnet_v2.resnet_v2_block('block2', base_depth=128, num_units=4, stride=2)],
            # num_classes=128,
            global_pool=False,
            is_training=is_training,
            reuse=False,
            include_root_block=False,
            scope='resnet_v2_101',
            post_norm_scope='postnorm_b2')
        net_c4, end_points = resnet_v2.resnet_v2(
            net_c3,
            blocks=[resnet_v2.resnet_v2_block('block3', base_depth=256, num_units=23, stride=2)],
            # num_classes=256,
            global_pool=False,
            is_training=is_training,
            reuse=False,
            include_root_block=False,
            scope='resnet_v2_101',
            post_norm_scope='postnorm_b3')
        net_c5, end_points = resnet_v2.resnet_v2(
            net_c4,
            blocks=[resnet_v2.resnet_v2_block('block4', base_depth=512, num_units=3, stride=2)],
            # num_classes=512,
            global_pool=False,
            is_training=is_training,
            reuse=False,
            include_root_block=False,
            scope='resnet_v2_101',
            post_norm_scope='postnorm_b4')
        print("net_c5:", net_c5.shape.as_list())
        print("net_c4:", net_c4.shape.as_list())
        print("net_c3:", net_c3.shape.as_list())
        print("net_c2:", net_c2.shape.as_list())
        print("net: ", net.shape.as_list())

        # top-down
        with tf.variable_scope("Encoder_resnet") as scope:
            net_p5 = slim.conv2d(net_c5, 256, kernel_size=1, stride=1, activation_fn=None)

            net_p4 = slim.conv2d(net_c4, 256, kernel_size=1, stride=1, activation_fn=None)
            # tensor_shape = net_p4.shape.as_list()
            net_p4 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(net_p5) + net_p4
            net_p4 = slim.conv2d(net_p4, 256, kernel_size=3, stride=1, activation_fn=None)

            net_p3 = slim.conv2d(net_c3, 256, kernel_size=1, stride=1, activation_fn=None)
            # tensor_shape = net_p3.shape.as_list()
            net_p3 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(net_p4) + net_p3
            net_p3 = slim.conv2d(net_p3, 256, kernel_size=3, stride=1, activation_fn=None)

            net_p2 = slim.conv2d(net_c2, 256, kernel_size=1, stride=1, activation_fn=None)
            # tensor_shape = net_p2.shape.as_list()
            net_p2 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(net_p3) + net_p2
            net_p2 = slim.conv2d(net_p2, 256, kernel_size=3, stride=1, activation_fn=None)
            # final conv+relu, then fcn
            net = slim.conv2d(net_p2, 64, kernel_size=3, stride=1)
            # net = tf.nn.relu(net)
            net = slim.conv2d(net, 1, kernel_size=3, stride=1)
            # net = tf.nn.relu(net)  # batch * 56 * 56 * 1
            net = slim.flatten(net) # flatten
            net = slim.fully_connected(net, 2048)

            # net, end_points = resnet_v2.resnet_v2_50(
            #     x,
            #     num_classes=None,
            #     is_training=is_training,
            #     reuse=reuse,
            #     scope='resnet_v2_50')
            # net = tf.squeeze(net, axis=[1, 2])
    variables = tf.contrib.framework.get_variables('resnet_v2_101')
    variables_top_down = tf.contrib.framework.get_variables(scope)
    variables.extend(variables_top_down)
    return net, variables

def Encoder_gru_dropout(x, initial_state, num_output=85, reuse = False):
    x_input = tf.expand_dims(x, 1)
    # print("gru_input:", x_input.shape.as_list())
    with tf.variable_scope("gru_dropout", reuse=reuse) as scope:
        gru_layer = tf.keras.layers.GRU(units=num_output, dropout=0.5)
        net = gru_layer(x_input, initial_state=initial_state)
    variables = tf.contrib.framework.get_variables(scope)
    return net, variables

def Encoder_fc3_dropout(x,
                        num_output=85,
                        is_training=True,
                        reuse=None,
                        name="3D_module"):
    """
    3D inference module. 3 MLP layers (last is the output)
    With dropout  on first 2.
    Input:
    - x: N x [|img_feat|, |3D_param|]
    - reuse: bool

    Outputs:
    - 3D params: N x num_output
      if orthogonal: 
           either 85: (3 + 24*3 + 10) or 109 (3 + 24*4 + 10) for factored axis-angle representation
      if perspective:
          86: (f, tx, ty, tz) + 24*3 + 10, or 110 for factored axis-angle.
    - variables: tf variables
    """
    if reuse:
        print('Reuse is on!')
    with tf.variable_scope(name, reuse=reuse) as scope:
        net = slim.fully_connected(x, 1024, scope='fc1')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout1')
        net = slim.fully_connected(net, 1024, scope='fc2')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout2')
        small_xavier = variance_scaling_initializer(
            factor=.01, mode='FAN_AVG', uniform=True)
        net = slim.fully_connected(
            net,
            num_output,
            activation_fn=None,
            weights_initializer=small_xavier,
            scope='fc3')

    variables = tf.contrib.framework.get_variables(scope)
    return net, variables


def get_encoder_fn_separate(model_type):
    """
    Retrieves diff encoder fn for image and 3D
    """
    encoder_fn = None
    threed_fn = None
    if 'resnet' in model_type:
        encoder_fn = Encoder_resnet
    else:
        print('Unknown encoder %s!' % model_type)
        exit(1)

    if 'fc3_dropout' in model_type:
        threed_fn = Encoder_fc3_dropout

    if encoder_fn is None or threed_fn is None:
        print('Dont know what encoder to use for %s' % model_type)
        with open("hmrlog.txt", 'a') as f:
            f.write("dont know what encoder to use..!!\n")
        exit(1)
        # import ipdb
        # ipdb.set_trace()

    return encoder_fn, threed_fn


def Discriminator_separable_rotations(
        poses,
        shapes,
        weight_decay,
):
    """
    23 Discriminators on each joint + 1 for all joints + 1 for shape.
    To share the params on rotations, this treats the 23 rotation matrices
    as a "vertical image":
    Do 1x1 conv, then send off to 23 independent classifiers.

    Input:
    - poses: N x 23 x 1 x 9, NHWC ALWAYS!!
    - shapes: N x 10
    - weight_decay: float

    Outputs:
    - prediction: N x (1+23) or N x (1+23+1) if do_joint is on.
    - variables: tf variables
    """
    data_format = "NHWC"
    with tf.name_scope("Discriminator_sep_rotations", values = [poses, shapes]):
        with tf.variable_scope("D") as scope:
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope([slim.conv2d], data_format=data_format):
                    poses = slim.conv2d(poses, 32, [1, 1], scope='D_conv1')
                    poses = slim.conv2d(poses, 32, [1, 1], scope='D_conv2')
                    theta_out = []
                    for i in range(0, 23):
                        theta_out.append(
                            slim.fully_connected(
                                poses[:, i, :, :],
                                1,
                                activation_fn=None,
                                scope="pose_out_j%d" % i))
                    theta_out_all = tf.squeeze(tf.stack(theta_out, axis=1))

                    # Do shape on it's own:
                    shapes = slim.stack(
                        shapes,
                        slim.fully_connected, [10, 5],
                        scope="shape_fc1")
                    shape_out = slim.fully_connected(
                        shapes, 1, activation_fn=None, scope="shape_final")
                    """ Compute joint correlation prior!"""
                    nz_feat = 1024
                    poses_all = slim.flatten(poses, scope='vectorize')
                    poses_all = slim.fully_connected(
                        poses_all, nz_feat, scope="D_alljoints_fc1")
                    poses_all = slim.fully_connected(
                        poses_all, nz_feat, scope="D_alljoints_fc2")
                    poses_all_out = slim.fully_connected(
                        poses_all,
                        1,
                        activation_fn=None,
                        scope="D_alljoints_out")
                    out = tf.concat([theta_out_all,
                                     poses_all_out, shape_out], 1)

            variables = tf.contrib.framework.get_variables(scope)
            return out, variables
