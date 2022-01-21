import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import os
import cv2
import sys
import time
from matplotlib import pyplot as plt
from tensorflow.contrib.framework.python.framework import checkpoint_utils
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
# print(os.path.join(BASE_DIR, 'models'))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))
import model_normalization as model
import data_sdf_h5_queue # as data
import output_utils
import create_file_lst
lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()

slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1', help='GPU to use [default: GPU 0]')
parser.add_argument('--category', type=str, default="chair", help='Which single class to train on [default: None]')
parser.add_argument('--log_dir', default='chair', help='Log dir [default: log]')
parser.add_argument('--num_points', type=int, default=1, help='Point Number [default: 2048]')
parser.add_argument("--beta1", type=float, dest="beta1",
                    default=0.5, help="beta1 of adams")
parser.add_argument('--num_sample_points', type=int, default=2048, help='Sample Point Number [default: 2048]')
# parser.add_argument('--sdf_points_num', type=int, default=32, help='Sample Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=20, help='Batch Size during training [default: 32]')
parser.add_argument('--img_h', type=int, default=138, help='Image Height')
parser.add_argument('--img_w', type=int, default=138, help='Image Width')
parser.add_argument('--sdf_res', type=int, default=256, help='sdf grid')
parser.add_argument('--num_classes', type=int, default=1024, help='vgg dim')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--restore_model', default='', help='restore_model') #checkpoint/sdf_2d3d_sdfbasic2_nowd
parser.add_argument('--restore_modelpn', default='', help='restore_model')#checkpoint/sdf_3dencoder_sdfbasic2/latest.ckpt
parser.add_argument('--restore_modelcnn', default='./models/CNN/pretrained_model/vgg_16.ckpt', help='restore_model')#../../models/CNN/pretrained_model/vgg_16.ckpt

parser.add_argument('--train_lst_dir', default=lst_dir, help='train mesh data list')
parser.add_argument('--valid_lst_dir', default=lst_dir, help='test mesh data list')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--mask_weight', type=float, default=4.0)
parser.add_argument('--threedcnn', action='store_true')
parser.add_argument('--volimp', action='store_true')
parser.add_argument('--img_feat_onestream', action='store_true')
parser.add_argument('--img_feat_twostream', action='store_true',default=True)
parser.add_argument('--binary', action='store_true',default=True)
parser.add_argument('--alpha', action='store_true')
parser.add_argument('--augcolorfore', action='store_true')
parser.add_argument('--augcolorback', action='store_true')
parser.add_argument('--backcolorwhite', action='store_true')
parser.add_argument('--rot', action='store_true')
parser.add_argument('--tanh', action='store_true')
parser.add_argument('--cam_est', action='store_true')
parser.add_argument('--cat_limit', type=int, default=168000, help="balance each category, 1500 * 24 = 36000")
parser.add_argument('--multi_view', action='store_true')

FLAGS = parser.parse_args()
print(FLAGS)

EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINTS = FLAGS.num_points
NUM_SAMPLE_POINTS = FLAGS.num_sample_points
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
PRETRAINED_MODEL_PATH = FLAGS.restore_model
PRETRAINED_CNN_MODEL_FILE = FLAGS.restore_modelcnn
PRETRAINED_PN_MODEL_FILE = FLAGS.restore_modelpn
LOG_DIR = FLAGS.log_dir


os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

RESULT_PATH = os.path.join(LOG_DIR, 'train_results')
if not os.path.exists(RESULT_PATH): os.mkdir(RESULT_PATH)

VALID_RESULT_PATH = os.path.join(LOG_DIR, 'valid_results')
if not os.path.exists(VALID_RESULT_PATH): os.mkdir(VALID_RESULT_PATH)

os.system('cp %s.py %s' % (os.path.splitext(model.__file__)[0], LOG_DIR))
os.system('cp train_sdf.py %s' % (LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
IMG_SIZE = 138
SDF_WEIGHT = 10.


TRAIN_LISTINFO = []
VAL_LISTINFO = []
cats_limit = {}
cats_limit_val = {}
cat_ids = []
if FLAGS.category == "all":
    for key, value in cats.items():
        cat_ids.append(value)
        cats_limit[value] = 0
        cats_limit_val[value] = 0
else:
    cat_ids.append(cats[FLAGS.category])
    cats_limit[cats[FLAGS.category]] = 0
    cats_limit_val[cats[FLAGS.category]] = 0

for cat_id in cat_ids:
    train_lst = os.path.join(FLAGS.train_lst_dir,"train_list_"+ cat_id+".txt")
    with open(train_lst, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            for render in range(12):
                cats_limit[cat_id]+=1
                TRAIN_LISTINFO += [(cat_id, line.strip(), render)]
                
info = {'rendered_dir': raw_dirs["renderedh5_dir"],
            'sdf_dir': raw_dirs["sdf_dir"]}

for cat_id in cat_ids:
    train_lst = os.path.join(lst_dir,"val_list_"+ cat_id+".txt")
    with open(train_lst, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            for render in range(12):
                cats_limit_val[cat_id]+=1
                VAL_LISTINFO += [(cat_id, line.strip(), render)]

TRAIN_DATASET = data_sdf_h5_queue.Pt_sdf_img(FLAGS, listinfo=TRAIN_LISTINFO, info=info, cats_limit=cats_limit)
VAL_DATASET = data_sdf_h5_queue.Pt_sdf_img(FLAGS, listinfo=VAL_LISTINFO, info=info, cats_limit=cats_limit_val)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 1e-6, name='lr') # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

class NoStdStreams(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()

def load_model(sess, LOAD_MODEL_FILE, prefixs, strict=False):

    vars_in_pretrained_model = dict(checkpoint_utils.list_variables(LOAD_MODEL_FILE))
    # print(vars_in_pretrained_model)
    vars_in_defined_model = []

    for var in tf.trainable_variables():
        if isinstance(prefixs, list):
            for prefix in prefixs:
                if (var.op.name.startswith(prefix)) and (var.op.name in vars_in_pretrained_model.keys()) and ('logits' not in var.op.name):
                    if (list(var.shape) == vars_in_pretrained_model[var.op.name]):
                        vars_in_defined_model.append(var)
        else:
            if (var.op.name.startswith(prefixs)) and (var.op.name in vars_in_pretrained_model.keys()) and ('logits' not in var.op.name):
                if (list(var.shape) == vars_in_pretrained_model[var.op.name]):
                    vars_in_defined_model.append(var)
    saver = tf.train.Saver(vars_in_defined_model)
    try:
        saver.restore(sess, LOAD_MODEL_FILE)
        print( "load_model Model loaded in file: %s" % (LOAD_MODEL_FILE))
    except:
        if strict:
            print( "Fail to load modelfile: %s" % LOAD_MODEL_FILE)
            return False
        else:
            print( "Fail loaded in file: %s" % (LOAD_MODEL_FILE))
            return True

    return True

def train():
    log_string(LOG_DIR)
    with tf.Graph().as_default():
        with tf.device('/gpu:1'):
            input_pls = model.placeholder_inputs(BATCH_SIZE, NUM_POINTS, (IMG_SIZE, IMG_SIZE),
                            num_sample_pc=NUM_SAMPLE_POINTS, scope='inputs_pl', FLAGS=FLAGS)
            
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0, name='batch')
            bn_decay = get_bn_decay(batch)
            # tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss

            end_points = model.get_model(input_pls, NUM_POINTS,is_training_pl, bn=False, FLAGS=FLAGS)
            loss, end_points = model.get_loss(end_points,
                sdf_weight=SDF_WEIGHT, mask_weight = FLAGS.mask_weight,
                                              num_sample_points=FLAGS.num_sample_points, FLAGS=FLAGS)
            # tf.summary.scalar('loss', loss)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta1)

            # Create a session
            config = tf.ConfigProto()
            gpu_options = tf.GPUOptions()#(per_process_gpu_memory_fraction=0.99)
            config=tf.ConfigProto(gpu_options=gpu_options)
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)

            merged = None
            train_writer = None

            ##### all
            update_variables = [x for x in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)]

            train_op = optimizer.minimize(loss, global_step=batch, var_list=update_variables)

            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)

            ######### Loading Checkpoint ###############
            # CNN(Pretrained from ImageNet)
            if PRETRAINED_CNN_MODEL_FILE is not '':
                if not load_model(sess, PRETRAINED_CNN_MODEL_FILE, 'vgg_16', strict=True):
                    return

            if PRETRAINED_PN_MODEL_FILE is not '':
                if not load_model(sess, PRETRAINED_PN_MODEL_FILE, ['refpc_reconstruction', 'sdfprediction'],
                                  strict=True):
                    return
                # Overall
            saver = tf.train.Saver([v for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES) if
                                    ('lr' not in v.name) and ('batch' not in v.name)])
            ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)

            if ckptstate is not None:
                LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
                try:
#                     load_model(sess, LOAD_MODEL_FILE, ['sdfprediction/fold1', 'sdfprediction/fold2','sdfprediction_imgfeat/fold1','sdfprediction_imgfeat/fold2','vgg_16'],
#                                strict=True)
                    load_model(sess, LOAD_MODEL_FILE, ['sdfprediction','sdfprediction_imgfeat/fold1','sdfprediction_imgfeat/fold2','vgg_16'], strict=True)
             
                    saver.restore(sess, LOAD_MODEL_FILE)
                    print("Model loaded in file: %s" % LOAD_MODEL_FILE)
                except:
                    print("Fail to load overall modelfile: %s" % PRETRAINED_MODEL_PATH)

            ###########################################

            ops = {'input_pls': input_pls,
                   'is_training_pl': is_training_pl,
                   'loss': loss,
                   'train_op': train_op,
                   'merged': merged,
                   'step': batch,
                   'lr': learning_rate,
                   'end_points': end_points}

            best_loss = 1e20
            TRAIN_DATASET.start()
            VAL_DATASET.start()
            best_acc = 0
            for epoch in range(MAX_EPOCH):
                log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()

                avg_accuracy = train_one_epoch(sess, ops, train_writer, saver)
                val_avg_accuracy = train_one_epoch_val(sess, ops, train_writer, saver)

                # Save the variables to disk.
                if val_avg_accuracy > best_acc:
                    best_acc = val_avg_accuracy
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                    log_string("best Model saved in file: %s" % save_path)
#                 elif epoch % 10 == 0:
#                     save_path = saver.save(sess, os.path.join(LOG_DIR, "model_epoch_%03d.ckpt"%(epoch)))
#                     log_string("Model saved in file: %s" % save_path)

            TRAIN_DATASET.shutdown()
            VAL_DATASET.shutdown()


def pc_normalize(pc, centroid=None):

    """ pc: NxC, return NxC """
    l = pc.shape[0]

    if centroid is None:
        centroid = np.mean(pc, axis=0)

    pc = pc - centroid
    # m = np.max(pc, axis=0)
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))

    pc = pc / m

    return pc
def train_one_epoch_val(sess, ops, train_writer, saver):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    num_batches = 200
#     num_batches = 2

    print('num_batches', num_batches)

    log_string(str(datetime.now()))

    loss_all = 0
    losses = {}
    for lossname in ops['end_points']['losses'].keys():
        losses[lossname] = 0

    tic = time.time()
    fetch_time = 0
    accuracy_epoch = 0
    for batch_idx in range(num_batches):
        start_fetch_tic = time.time()
        batch_data = VAL_DATASET.fetch()
        fetch_time += (time.time() - start_fetch_tic)
        feed_dict = {ops['is_training_pl']: is_training,
                     ops['input_pls']['voxel_pt']: batch_data['voxel_pt'],
                     ops['input_pls']['voxel_val']: batch_data['voxel_val'],
                     ops['input_pls']['occupied_pt']: batch_data['occupied_pt'],
                     ops['input_pls']['imgs']: batch_data['img'],
                     ops['input_pls']['trans_mat']: batch_data['trans_mat']}
        output_list = [ops['train_op'], ops['step'], ops['lr'], ops['loss'], ops['end_points']['pred_sdf'],
                       ops['end_points']['ref_sdf'], ops['end_points']['sample_img_points'],
                       ops['end_points']['pred_sdf'], ops['end_points']['ref_img']]

        loss_list = []
        for il, lossname in enumerate(losses.keys()):
            loss_list += [ops['end_points']['losses'][lossname]]

        outputs = sess.run(output_list + loss_list, feed_dict=feed_dict)

        _, step, lr_val, loss_val, pred_sdf_val, ref_sdf_val, \
        sample_img_points_val, pred_sdf_val, ref_img_val  = outputs[:-len(losses)]

        for il, lossname in enumerate(losses.keys()):
            if lossname == "accuracy":
                accuracy_epoch += outputs[len(output_list)+il]
            losses[lossname] += outputs[len(output_list)+il]

    outstr = "avg val accuracy: %f"% (accuracy_epoch / num_batches)
    log_string(outstr)
#     print("avg accuracy:", accuracy_epoch / num_batches)
    return accuracy_epoch / num_batches

def train_one_epoch(sess, ops, train_writer, saver):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    num_batches = int(len(TRAIN_DATASET) / BATCH_SIZE)
#     num_batches = 2 
    print('num_batches', num_batches)

    log_string(str(datetime.now()))

    loss_all = 0
    losses = {}
    for lossname in ops['end_points']['losses'].keys():
        losses[lossname] = 0

    tic = time.time()
    fetch_time = 0
    accuracy_epoch = 0
    for batch_idx in range(num_batches):
        start_fetch_tic = time.time()
        batch_data = TRAIN_DATASET.fetch()
        fetch_time += (time.time() - start_fetch_tic)
        feed_dict = {ops['is_training_pl']: is_training,
                     ops['input_pls']['voxel_pt']: batch_data['voxel_pt'],
                     ops['input_pls']['voxel_val']: batch_data['voxel_val'],
                     ops['input_pls']['occupied_pt']: batch_data['occupied_pt'],
                     ops['input_pls']['imgs']: batch_data['img'],
                     ops['input_pls']['trans_mat']: batch_data['trans_mat']}
        output_list = [ops['train_op'], ops['step'], ops['lr'], ops['loss'], ops['end_points']['pred_sdf'],
                       ops['end_points']['ref_sdf'], ops['end_points']['sample_img_points'],
                       ops['end_points']['pred_sdf'], ops['end_points']['ref_img']]

        loss_list = []
        for il, lossname in enumerate(losses.keys()):
            loss_list += [ops['end_points']['losses'][lossname]]

        outputs = sess.run(output_list + loss_list, feed_dict=feed_dict)

        _, step, lr_val, loss_val, pred_sdf_val, ref_sdf_val, \
        sample_img_points_val, pred_sdf_val, ref_img_val = outputs[:-len(losses)]
#   
#         pred_sdf_val /= SDF_WEIGHT
#         np.savetxt(os.path.join(RESULT_PATH, '%d_class_pred.txt' % batch_idx), classes)
#         np.savetxt(os.path.join(RESULT_PATH, '%d_middle_pred.txt' % batch_idx), middle[0,:,:])
#         np.savetxt(os.path.join(RESULT_PATH, '%d_pred.txt' % batch_idx), pred_sdf_val[0,:,:])
#         np.savetxt(os.path.join(RESULT_PATH, '%d_ref_sdf.txt' % batch_idx), ref_sdf_val[0,:,:])

        for il, lossname in enumerate(losses.keys()):
            if lossname == "accuracy":
                accuracy_epoch += outputs[len(output_list)+il]
            losses[lossname] += outputs[len(output_list)+il]

        loss_all += losses['overall_loss']

        verbose_freq = 20.
        if (batch_idx + 1) % verbose_freq == 0:
            bid = 0
            # sampling
            if (batch_idx + 1) % (20*verbose_freq) == 0:
                saveimg = ref_img_val[bid, :, :, :] 
                samplept_img = sample_img_points_val[bid, ...]
                choice = np.random.randint(samplept_img.shape[0], size=100)
                samplept_img = samplept_img[choice, ...]
            outstr = ' -- %03d / %03d -- ' % (batch_idx+1, num_batches)
            for lossname in losses.keys():
                outstr += '%s: %f, ' % (lossname, losses[lossname] / verbose_freq)
                losses[lossname] = 0
            outstr += "lr: %f" % (lr_val)
            outstr += ' time: %.02f, ' % (time.time() - tic)
            outstr += ', fetch time per b: %.02f, ' % (fetch_time/verbose_freq)
            tic = time.time()
            fetch_time = 0
            log_string(outstr)
    log_string("avg accuracy: %f"% (accuracy_epoch / num_batches))
    return accuracy_epoch / num_batches

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
