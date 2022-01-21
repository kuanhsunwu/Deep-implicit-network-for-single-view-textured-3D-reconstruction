import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import vgg
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR,'data'))
sys.path.append(os.path.join(BASE_DIR, 'models'))
print(os.path.join(BASE_DIR, 'models'))
import sdfnet

from sklearn.neighbors import KDTree

def placeholder_inputs(batch_size, num_points, img_size, num_sample_pc = 256, scope='', FLAGS=None):


    with tf.variable_scope(scope) as sc:
        
        voxel_pt = tf.placeholder(tf.float32, shape=(batch_size, num_sample_pc, 3))
        voxel_val = tf.placeholder(tf.float32, shape=(batch_size, num_sample_pc, 3))
        occupied_pt = tf.placeholder(tf.float32, shape=(batch_size, num_sample_pc, 3))
        if FLAGS.alpha:
            imgs_pl = tf.placeholder(tf.float32, shape=(batch_size, img_size[0], img_size[1], 4))
        else:
            imgs_pl = tf.placeholder(tf.float32, shape=(batch_size, img_size[0], img_size[1], 3))
       
        trans_mat_pl = tf.placeholder(tf.float32, shape=(batch_size, 4, 3))
    sdf = {}
    sdf['voxel_pt'] = voxel_pt
    sdf['voxel_val'] = voxel_val
    sdf['occupied_pt'] = occupied_pt
    sdf['imgs'] = imgs_pl
    sdf['trans_mat'] = trans_mat_pl
    return sdf


def placeholder_features(batch_size, num_sample_pc = 256, scope=''):
    with tf.variable_scope(scope) as sc:
        ref_feats_embedding_cnn_pl = tf.placeholder(tf.float32, shape=(batch_size, 1, 1, 1024))
        point_img_feat_pl = tf.placeholder(tf.float32, shape=(batch_size, num_sample_pc, 1, 1472))
    feat = {}
    feat['ref_feats_embedding_cnn'] = ref_feats_embedding_cnn_pl
    feat['point_img_feat'] = point_img_feat_pl
    return feat

def get_model(ref_dict, num_point, is_training, bn=False, bn_decay=None, img_size = 224, wd=1e-5, FLAGS=None):

    ref_img = ref_dict['imgs']
    ref_voxel_pt = ref_dict['voxel_pt']
    ref_voxel_val = ref_dict['voxel_val']
    ref_occupied_pt = ref_dict['occupied_pt']
    ref_trans_mat = ref_dict['trans_mat']

    batch_size = ref_img.get_shape()[0].value

    # endpoints
    end_points = {}
    end_points['ref_voxel_pt'] = ref_voxel_pt
    end_points['ref_voxel_val'] = ref_voxel_val
    end_points['ref_img'] = ref_img # B*H*W*3|4

    # Image extract features
    if ref_img.shape[1] != img_size or ref_img.shape[2] != img_size:
        if FLAGS.alpha:
            ref_img_rgb = tf.image.resize_bilinear(ref_img[:,:,:,:3], [img_size, img_size])
            ref_img_alpha = tf.image.resize_nearest_neighbor(
                tf.expand_dims(ref_img[:,:,:,3], axis=-1), [img_size, img_size])
            ref_img = tf.concat([ref_img_rgb, ref_img_alpha], axis = -1)
        else:
            ref_img = tf.image.resize_bilinear(ref_img, [img_size, img_size])
    end_points['resized_ref_img'] = ref_img
    vgg.vgg_16.default_image_size = img_size
    with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(wd)):
        ref_feats_embedding, vgg_end_points = vgg.vgg_16(ref_img, num_classes=FLAGS.num_classes, is_training=False, scope='vgg_16', spatial_squeeze=False)
        ref_feats_embedding_cnn = tf.squeeze(ref_feats_embedding, axis = [1,2])
    end_points['img_embedding'] = ref_feats_embedding_cnn
    point_img_feat=None
    pred_sdf=None
    if FLAGS.binary:
       
         ## -------------------------my modification ---------------------------------------------##           
        if FLAGS.img_feat_twostream:
            sample_img_points = get_img_points(ref_occupied_pt, ref_trans_mat)  # B * N * 2
            vgg_conv1 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv1/conv1_2'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv1 = tf.contrib.resampler.resampler(vgg_conv1, sample_img_points)
            print('point_vgg_conv1', point_vgg_conv1.shape)
            vgg_conv2 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv2/conv2_2'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv2 = tf.contrib.resampler.resampler(vgg_conv2, sample_img_points)
            print('point_vgg_conv2', point_vgg_conv2.shape)
            vgg_conv3 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv3/conv3_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv3 = tf.contrib.resampler.resampler(vgg_conv3, sample_img_points)
            print('point_vgg_conv3', point_vgg_conv3.shape)
            vgg_conv4 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv4/conv4_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv4 = tf.contrib.resampler.resampler(vgg_conv4, sample_img_points)
            print('point_vgg_conv4', point_vgg_conv4.shape)
            vgg_conv5 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv5/conv5_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv5 = tf.contrib.resampler.resampler(vgg_conv5, sample_img_points)
            print('point_vgg_conv5', point_vgg_conv5.shape)

            point_img_feat = tf.concat(axis=2,
               values=[point_vgg_conv1, point_vgg_conv2, point_vgg_conv3, point_vgg_conv4,
                       point_vgg_conv5])
            point_img_feat = tf.expand_dims(point_img_feat, axis=2)
            print('point_img_feat', point_img_feat.shape)
            if not FLAGS.multi_view:
                # Predict SDF
                with tf.variable_scope("sdfprediction") as scope:
                    pred_sdf_value_global, pred_label_global  = sdfnet.get_sdf_basic2_binary(ref_voxel_pt, ref_feats_embedding_cnn, is_training, batch_size, num_point, bn, bn_decay, wd=wd)

                with tf.variable_scope("sdfprediction_imgfeat") as scope:
                    pred_sdf_value_local, pred_label_local = sdfnet.get_sdf_basic2_imgfeat_twostream_binary(ref_voxel_pt, point_img_feat, is_training, batch_size, num_point,bn, bn_decay, wd=wd)
                    
                    
                pred_sdf = pred_sdf_value_global + pred_sdf_value_local
                pred_label = pred_label_global + pred_label_local
#                 print('pred_sdf', pred_sdf.shape)
                
             ## -------------------------my modification ---------------------------------------------##        
        else:
            # sample_img_points = tf.zeros((batch_size, FLAGS.num_sample_points, 2), dtype=tf.float32)
            sample_img_points = get_img_points(ref_sample_pc, ref_trans_mat)
            if not FLAGS.multi_view:
                with tf.variable_scope("sdfprediction") as scope:
                    pred_sdf = sdfnet.get_sdf_basic2_binary(ref_sample_pc, ref_feats_embedding_cnn,
                                is_training, batch_size, num_point, bn, bn_decay,wd=wd)
    else:
       
        if FLAGS.img_feat_twostream:
            sample_img_points = get_img_points(ref_sample_pc, ref_trans_mat)  # B * N * 2
            vgg_conv1 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv1/conv1_2'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv1 = tf.contrib.resampler.resampler(vgg_conv1, sample_img_points)
            print('point_vgg_conv1', point_vgg_conv1.shape)
            vgg_conv2 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv2/conv2_2'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv2 = tf.contrib.resampler.resampler(vgg_conv2, sample_img_points)
            print('point_vgg_conv2', point_vgg_conv2.shape)
            vgg_conv3 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv3/conv3_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv3 = tf.contrib.resampler.resampler(vgg_conv3, sample_img_points)
            print('point_vgg_conv3', point_vgg_conv3.shape)
            vgg_conv4 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv4/conv4_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv4 = tf.contrib.resampler.resampler(vgg_conv4, sample_img_points)
            print('point_vgg_conv4', point_vgg_conv4.shape)
            vgg_conv5 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv5/conv5_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv5 = tf.contrib.resampler.resampler(vgg_conv5, sample_img_points)
            print('point_vgg_conv5', point_vgg_conv5.shape)

            point_img_feat = tf.concat(axis=2,
                                       values=[point_vgg_conv1, point_vgg_conv2, point_vgg_conv3, point_vgg_conv4,
                                               point_vgg_conv5])
            point_img_feat = tf.expand_dims(point_img_feat, axis=2)
            print('point_img_feat', point_img_feat.shape)
            if not FLAGS.multi_view:
                # Predict SDF
                with tf.variable_scope("sdfprediction") as scope:
                    pred_sdf_value_global = sdfnet.get_sdf_basic2(ref_sample_pc_rot, ref_feats_embedding_cnn,
                                                                     is_training, batch_size, num_point, bn, bn_decay,
                                                                     wd=wd)

                with tf.variable_scope("sdfprediction_imgfeat") as scope:
                    pred_sdf_value_local = sdfnet.get_sdf_basic2_imgfeat_twostream(ref_sample_pc_rot, point_img_feat,
                                                                                      is_training, batch_size, num_point,
                                                                                      bn, bn_decay, wd=wd)

                pred_sdf = pred_sdf_value_global + pred_sdf_value_local
                end_points["pred_sdf_value_global"] = pred_sdf_value_global
                end_points["pred_sdf_value_local"] = pred_sdf_value_local
        else:
            # sample_img_points = tf.zeros((batch_size, FLAGS.num_sample_points, 2), dtype=tf.float32)
            sample_img_points = get_img_points(ref_sample_pc, ref_trans_mat)
            if not FLAGS.multi_view:
                with tf.variable_scope("sdfprediction") as scope:
                    pred_sdf = sdfnet.get_sdf_basic2(ref_sample_pc, ref_feats_embedding_cnn,
                                is_training, batch_size, num_point, bn, bn_decay,wd=wd)
        if FLAGS.tanh:
            pred_sdf = tf.tanh(pred_sdf)
    end_points['pred_label'] = pred_label       
    end_points['pred_sdf'] = pred_sdf
    end_points["sample_img_points"] = sample_img_points
    end_points["ref_feats_embedding_cnn"] = ref_feats_embedding_cnn
    end_points["point_img_feat"] = point_img_feat

    return end_points

def get_decoder(num_point, input_pls, feature_pls, bn=False, bn_decay=None,wd=None):
    ref_feats_embedding_cnn = feature_pls["ref_feats_embedding_cnn"]
    point_img_feat = feature_pls["point_img_feat"]
    ref_sample_pc_rot = input_pls['sample_pc_rot']

    with tf.variable_scope("sdfprediction") as scope:
        pred_sdf_value_global = sdfnet.get_sdf_basic2(ref_sample_pc_rot, ref_feats_embedding_cnn,
                                                         False, 1, num_point, bn, bn_decay,
                                                         wd=wd)

    with tf.variable_scope("sdfprediction_imgfeat") as scope:
        pred_sdf_value_local = sdfnet.get_sdf_basic2_imgfeat_twostream(ref_sample_pc_rot, point_img_feat,
                                                                          False, 1, num_point,
                                                                          bn, bn_decay, wd=wd)
    multi_pred_sdf = pred_sdf_value_global + pred_sdf_value_local
    return multi_pred_sdf


def get_img_points(sample_pc, trans_mat_right):
    # sample_pc B*N*3
    size_lst = sample_pc.get_shape().as_list()
    homo_pc = tf.concat((sample_pc, tf.ones((size_lst[0], size_lst[1], 1),dtype=np.float32)),axis= -1)
    print("homo_pc.get_shape()", homo_pc.get_shape())
    pc_xyz = tf.matmul(homo_pc, trans_mat_right)
    print("pc_xyz.get_shape()", pc_xyz.get_shape()) # B * N * 3
    pc_xy = tf.divide(pc_xyz[:,:,:2], tf.expand_dims(pc_xyz[:,:,2], axis = 2))
    mintensor = tf.constant([0.0,0.0], dtype=tf.float32)
    maxtensor = tf.constant([137.0,137.0], dtype=tf.float32)
    return tf.minimum(maxtensor, tf.maximum(mintensor, pc_xy))


def get_loss(end_points, sdf_weight=10., regularization=True, mask_weight = 4.,
             num_sample_points = 2048, FLAGS=None, batch_size=None):

    pred_sdf = end_points['pred_sdf']
    pred_label = end_points['pred_label']
    ref_voxel_val = end_points['ref_voxel_val']
    ################
    # Compute loss #
    ################
    end_points['losses'] = {}
    end_points['ref_sdf'] = ref_voxel_val


    if batch_size is None:
        batch_size = FLAGS.batch_size
    if FLAGS.binary:
       
        first_column_val = tf.gather(ref_voxel_val, tf.constant([0]), axis=2)
        label_sdf = tf.reshape(tf.cast(tf.math.greater(first_column_val, tf.constant(-1.0)), dtype=tf.int32),
                               (batch_size, num_sample_points))
        
        label_accuracy = tf.reduce_mean(tf.cast(tf.equal(label_sdf, tf.argmax(pred_label, axis=2, output_type=tf.int32)), dtype=tf.float32))
        end_points['losses']['accuracy'] = label_accuracy
        sdf_label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_sdf, logits=pred_label)
        sdf_label_loss = tf.reduce_mean(sdf_label_loss) * 100

        
        pred_label_digit = tf.argmax(pred_label, axis=2, output_type=tf.int32)
        end_points['class'] = pred_label_digit
        pred_label_digit = tf.where(tf.equal(pred_label_digit, 0), tf.constant(-1.0, shape=pred_label_digit.shape)  , tf.constant(0.0, shape=pred_label_digit.shape))
        
        pred_label_digit = tf.stack([pred_label_digit, pred_label_digit, pred_label_digit],axis=2)
        print('pred_label_digit',pred_label_digit.shape)
        end_points['middle'] = pred_label_digit
        pred_sdf = tf.where(tf.equal(pred_label_digit, -1),pred_label_digit,pred_sdf)
        end_points['pred_sdf'] = pred_sdf
     



        sdf_colour_loss = tf.reduce_mean(tf.abs(pred_sdf - ref_voxel_val)) * 255
        sdf_loss = sdf_label_loss + sdf_colour_loss
        end_points['losses']['sdf_label_loss'] = sdf_label_loss
        end_points['losses']['sdf_colour_loss'] = sdf_colour_loss
        end_points['losses']['sdf_loss'] = sdf_loss
    else:
        ############### accuracy
        zero = tf.constant(0, dtype=tf.float32)
        gt_sign = tf.greater(gt_sdf,zero)
        pred_sign = tf.greater(pred_sdf,zero)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(gt_sign, pred_sign), dtype=tf.float32))
        end_points['losses']['accuracy'] = accuracy
        weight_mask = tf.to_float(tf.less_equal(gt_sdf, tf.constant(0.01))) * mask_weight \
                      + tf.to_float(tf.greater(gt_sdf, tf.constant(0.01)))
        end_points['weighed_mask'] = weight_mask
        sdf_loss = tf.reduce_mean(tf.abs(gt_sdf * sdf_weight - pred_sdf) * weight_mask)
        end_points['losses']['sdf_loss_realvalue'] = tf.reduce_mean(tf.abs(gt_sdf - pred_sdf / sdf_weight))
        sdf_loss = sdf_loss * 1000
    end_points['losses']['sdf_loss'] = sdf_loss
    loss = sdf_loss
    ############### weight decay
    if regularization:
        vgg_regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
        decoder_regularization_loss = tf.add_n(tf.get_collection('regularizer'))
        end_points['losses']['regularization'] = (vgg_regularization_loss + decoder_regularization_loss)
        loss += (vgg_regularization_loss + decoder_regularization_loss)
    end_points['losses']['overall_loss'] = loss
    return loss, end_points
