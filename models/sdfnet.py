import tensorflow as tf
import tf_util


def get_sdf_basic2_imgfeat_twostream_binary(src_pc, point_feat, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    net2 = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    concat = tf.concat(axis=3, values=[net2, point_feat])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')
    pred_label_local_ = tf_util.conv2d(net2, 2, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5_bi')
    
    pred_label_local = tf.reshape(pred_label_local_, [batch_size, -1, 2])

    
    concat_colour = tf.concat(axis=3, values=[pred_label_local_, concat])

    net3 = tf_util.conv2d(concat_colour, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold3/conv1')
    net3 = tf_util.conv2d(net3, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold3/conv2')
    pred_sdf_value_local = tf_util.conv2d(net3, 3, [1,1], padding='VALID', stride=[1,1], activation_fn=tf.sigmoid, bn=False, weight_decay=wd, scope='fold2/conv5_colour')

    pred_sdf_value_local = tf.reshape(pred_sdf_value_local, [batch_size, -1, 3])
    
    return pred_sdf_value_local, pred_label_local

def get_sdf_basic2_binary(src_pc, globalfeats, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net2 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    net2 = tf_util.conv2d(net2, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
    globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
    print( 'net2', net2.shape)
    print( 'globalfeats_expand', globalfeats_expand.shape)
    concat = tf.concat(axis=3, values=[net2, globalfeats_expand])

    net2 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')
    pred_label_local_ = tf_util.conv2d(net2, 2, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='fold2/conv5_bi')

    pred_label_local = tf.reshape(pred_label_local_, [batch_size, -1, 2])
    
    
    concat_colour = tf.concat(axis=3, values=[pred_label_local_, concat])

    net3 = tf_util.conv2d(concat_colour, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold3/conv1')
    net3 = tf_util.conv2d(net3, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold3/conv2')
    pred_sdf_value_local = tf_util.conv2d(net3, 3, [1,1], padding='VALID', stride=[1,1], activation_fn=tf.sigmoid, bn=False, weight_decay=wd, scope='fold2/conv5_colour')

    pred_sdf_value_local = tf.reshape(pred_sdf_value_local, [batch_size, -1, 3])

    return pred_sdf_value_local, pred_label_local
