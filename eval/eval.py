import argparse
from datetime import datetime
import numpy as np
import random
import tensorflow as tf
import os
import sys
import h5py
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))
import model_normalization as model
import create_file_lst
import cv2
import output_utils
import data_sdf_h5_queue
from matplotlib import pyplot as plt

slim = tf.contrib.slim
lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 201]')
parser.add_argument('--img_h', type=int, default=138, help='Image Height')
parser.add_argument('--img_w', type=int, default=138, help='Image Width')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--num_classes', type=int, default=1024, help='vgg dim')
parser.add_argument('--num_points', type=int, default=1, help='Point Number [default: 2048]')
parser.add_argument('--sdf_res', type=int, default=64, help='sdf grid')
parser.add_argument('--alpha', action='store_true')
parser.add_argument('--rot', action='store_true')
parser.add_argument('--tanh', action='store_true')
parser.add_argument('--cat_limit', type=int, default=168000, help="balance each category, 1500 * 24 = 36000")
parser.add_argument('--multi_view', action='store_true')
parser.add_argument('--num_sample_points', type=int, default=1, help='Sample Point Number [default: 2048]')
parser.add_argument('--shift', action="store_true")
parser.add_argument('--loss_mode', type=str, default="3D", help='loss on 3D points or 2D points')

parser.add_argument('--log_dir', default='/media/sdb/kuanhsun/one_chair', help='Log dir [default: log]')
parser.add_argument('--restore_model', default='./chair/', help='restore_model') #checkpoint/sdf_2d3d_sdfbasic2_nowd

parser.add_argument('--cam_log_dir', default='./cam_est/checkpoint/cam_DISN', help='Log dir [default: log]')
parser.add_argument('--test_lst_dir', default=lst_dir, help='test mesh data list')
parser.add_argument('--iso', type=float, default=0.0, help='iso value')
parser.add_argument('--threedcnn', action='store_true')
parser.add_argument('--img_feat_onestream', action='store_true')
parser.add_argument('--img_feat_twostream', action='store_true',default=True)
parser.add_argument('--category', default="chair", help='Which single class to train on [default: None]')
parser.add_argument('--binary', action='store_true',default=True)
parser.add_argument('--create_obj', action='store_true', help="create_obj or test accuracy on test set")
parser.add_argument('--store', action='store_true')
parser.add_argument('--view_num', type=int, default=24, help="how many views do you want to create for each obj")
parser.add_argument('--cam_est', action='store_true', help="if you are using the estimated camera image h5")

parser.add_argument('--augcolorfore', action='store_true')
parser.add_argument('--augcolorback', action='store_true')
parser.add_argument('--backcolorwhite', action='store_true')

FLAGS = parser.parse_args()
print(FLAGS)

NUM_POINTS = FLAGS.num_points
BATCH_SIZE = FLAGS.batch_size
RESOLUTION = FLAGS.sdf_res
TOTAL_POINTS = RESOLUTION * RESOLUTION * RESOLUTION


# NUM_SAMPLE_POINTS = 65536  * 2
NUM_SAMPLE_POINTS = 65536 * 2 
GPU_INDEX = FLAGS.gpu
PRETRAINED_MODEL_PATH = FLAGS.restore_model
LOG_DIR = FLAGS.log_dir

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

IMG_SIZE = FLAGS.img_h


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)



TEST_LISTINFO = []
cats_limit = {}
# print(FLAGS)


lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()

cat_ids = []
if FLAGS.category == "all":
    for key, value in cats.items():
        cat_ids.append(value)
        cats_limit[value] = 0
else:
    cat_ids.append(cats[FLAGS.category])
    cats_limit[cats[FLAGS.category]] = 0

for cat_id in cat_ids:
    train_lst = os.path.join(lst_dir,"test_list_"+ cat_id+".txt")
    with open(train_lst, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            for render in range(12):
                cats_limit[cat_id]+=1
                TEST_LISTINFO += [(cat_id, line.strip(), render)]
# for render in range(12):
#     TEST_LISTINFO += [('02958343', '6f1888091dcf4ca550577cf04f3bf74a', render)]   
#     cats_limit['02958343']+=1  
# for render in range(12):
#     TEST_LISTINFO += [('02958343', '1eae4fcec701c176e742b0b5e87bec54', render)]   
#     cats_limit['02958343']+=1   
# for render in range(12):
#     cats_limit['04379243']+=1 
#     TEST_LISTINFO += [('04379243', '2ba8eb5ec0a05694593ebeeedbff73b', render)]        
# for render in range(12):
#     cats_limit['04379243']+=1 
#     TEST_LISTINFO += [('04379243', '39f202c92b0afd0429d8eecf3827c486', render)]        
# for render in range(12):
#     cats_limit['03001627']+=1 
#     TEST_LISTINFO += [('03001627', '53675c4bbb33fe72bcc0c5df96f9b28e', render)]        
# for render in range(12):
#     cats_limit['03001627']+=1 
#     TEST_LISTINFO += [('03001627', '4ab439279e665e08410fc47639efb60', render)]        
# for render in range(12):
#     cats_limit['03467517']+=1 
#     TEST_LISTINFO += [('03467517', 'b54b9a0eb5be15dd57700c05b1862d8', render)]        
# for render in range(12):
#     cats_limit['03467517']+=1 
#     TEST_LISTINFO += [('03467517', '1e56954ca36bbfdd6b05157edf4233a3', render)] 
# print(TEST_LISTINFO)
# TEST_LISTINFO += [('02958343', '81bb9f249771d31ac675ce607c2b4b5f', 11)]   

# TEST_LISTINFO += [('02958343', '5b1f078a81df862ec2c2c81e2232aa95', 11)]   
# TEST_LISTINFO += [('02958343', '2b5a333c1a5aede3b5449ea1678de914', 11)]   
# cats_limit['02958343']+=1
info = {'rendered_dir': raw_dirs["renderedh5_dir"],
            'sdf_dir': raw_dirs["sdf_dir"]}
TEST_DATASET = data_sdf_h5_queue.Pt_sdf_img(FLAGS, listinfo=TEST_LISTINFO, info=info, cats_limit=cats_limit,shuffle=False)

# TEST_DATASET.start()
# batch_data = TEST_DATASET.fetch()
# print(batch_data['trans_mat'].shape)
# TEST_DATASET.shutdown()
def create():
    with tf.device('/gpu:1'):
        input_pls = model.placeholder_inputs(BATCH_SIZE, NUM_POINTS, (IMG_SIZE, IMG_SIZE),
                            num_sample_pc=NUM_SAMPLE_POINTS, scope='inputs_pl', FLAGS=FLAGS)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        batch = tf.Variable(0, name='batch')

        print("--- Get model and loss")
        # Get model and loss

        end_points = model.get_model(input_pls, NUM_POINTS, is_training_pl, bn=False,FLAGS=FLAGS)

        loss, end_points = model.get_loss(end_points,
            sdf_weight=10, num_sample_points=NUM_SAMPLE_POINTS, FLAGS=FLAGS)
        # Create a session
        gpu_options = tf.GPUOptions() # per_process_gpu_memory_fraction=0.99
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
    
        sess = tf.Session(config=config)

        init = tf.global_variables_initializer()
        sess.run(init)

        ######### Loading Checkpoint ###############
        saver = tf.train.Saver([v for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES) if
                                ('lr' not in v.name) and ('batch' not in v.name)])
        ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)
        if ckptstate is not None:
            LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
            try:
                # load_model(sess, PRETRAINED_PN_MODEL_FILE, ['refpc_reconstruction','sdfprediction','vgg_16'], strict=True)

                saver.restore(sess, LOAD_MODEL_FILE)
                print("Model loaded in file: %s" % LOAD_MODEL_FILE)
            except:
                print("Fail to load overall modelfile: %s" % PRETRAINED_MODEL_PATH)

        ###########################################

        ops = {'input_pls': input_pls,
               'is_training_pl': is_training_pl,
               'loss': loss,
               'step': batch,
               'end_points': end_points}

        test_one_epoch(sess, ops)



def metric(point_gt, point_pred, color_gt, color_pred):
    from scipy.spatial.distance import cdist

    dis_mat = cdist(point_gt, point_pred) 
    min_dis = np.amin(dis_mat, axis=1)
    point_iou = min_dis[min_dis < 1].shape[0]/(point_gt.shape[0] + point_pred.shape[0] - min_dis[min_dis < 1].shape[0])
    
    index_pre = np.argmin(dis_mat, axis=1)
    mse_rgb_tmp = np.sum(np.square(color_gt - color_pred[index_pre]), axis=0)
    avg_mse_rgb_tmp = np.sum(mse_rgb_tmp) / index_pre.shape[0] / 3
    avg_psnr_rgb_tmp = 20*np.log10(1) - 10*np.log10(avg_mse_rgb_tmp)
    
    return point_iou, avg_psnr_rgb_tmp


def output_voxel_color_point_cloud(data, out_file):
    with open(out_file, 'w') as f:
        pre_voxel = data[:,3] 
    
        data = data[np.where(pre_voxel != -1.0)]
        color = data[:,3:]
        color[color > 1] = 1
        color[color < 0] = 0
        data[:,3:] = color
        for i in range(data.shape[0]):
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], data[i][3]*255, data[i][4]*255, data[i][5]*255))
    return data         
            
def output_point_cloud(data, out_file):
    with open(out_file, 'w') as f:

        for i in range(data.shape[0]):
            f.write('%f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5]))            



def test_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    # Shuffle train samples
    losses = {}
    for lossname in ops['end_points']['losses'].keys():
        losses[lossname] = 0
        
    save_dir = os.path.join(LOG_DIR,'chair')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
        
       
    db_dir = raw_dirs['renderedh5_dir'] 
    TEST_DATASET.start()
    
    result_iou = []
    result_psnr = []
    pre_id , _obj, _num = TEST_LISTINFO[0]
    print('TEST_DATASET %d'%len(TEST_DATASET))
    for i in range(len(TEST_DATASET)):
        
        cat_id, obj, num = TEST_LISTINFO[i]
            
        if (i % 12) == 0 :
            
            
                
                
            save_obj_dir = os.path.join(save_dir,cat_id,obj)
            if not os.path.exists(save_obj_dir):
                os.makedirs(save_obj_dir)
                    
            if not i==0 :    
                result_iou.append(obj_iou)
                result_psnr.append(obj_psnr)
            obj_iou = []
            obj_psnr =[]
                
            if cat_id != pre_id :
                np.save(os.path.join(save_dir,pre_id,'result_iou'+ pre_id), result_iou)
                np.save(os.path.join(save_dir,pre_id,'result_psnr' +pre_id), result_psnr)
                result_iou = []
                result_psnr = []
                    
        
        
        
        batch_data = TEST_DATASET.fetch()

    
    
        x_ = np.linspace(-32, 31, num=RESOLUTION)
        y_ = np.linspace(-32, 31, num=RESOLUTION)
        z_ = np.linspace(-32, 31, num=RESOLUTION)
        z, y, x = np.meshgrid(z_, y_, x_, indexing='ij')
        x = np.expand_dims(x, 3)
        y = np.expand_dims(y, 3)
        z = np.expand_dims(z, 3)
        all_pts = np.concatenate((x, y, z), axis=3).astype(np.float32)
        all_pts = all_pts.reshape(1, -1, 3)
            

        save_preds = np.empty((1,6), float)
        for j in range(int(all_pts.shape[1]/(NUM_SAMPLE_POINTS))):
                
            _pt = all_pts[:, j*NUM_SAMPLE_POINTS : (j+1)*NUM_SAMPLE_POINTS ,:]
            
            feed_dict = {ops['is_training_pl']: is_training,
                         ops['input_pls']['voxel_pt']: _pt,
                         ops['input_pls']['occupied_pt']: _pt,
                         ops['input_pls']['imgs']: batch_data['img'].reshape(BATCH_SIZE, 138, 138, 3),
                         ops['input_pls']['trans_mat']: batch_data['trans_mat'].reshape(BATCH_SIZE, 4, 3)}
            
            
            output_list = [ops['end_points']['pred_sdf'], ops['end_points']['ref_img'],
                           ops['end_points']['sample_img_points']]
            pred_sdf_val, ref_img_val, sample_img_points_val = sess.run(output_list, feed_dict=feed_dict)
            
            
            
            save_ref = np.concatenate((_pt[0,:,:], pred_sdf_val[0,:,:]), axis=1)
            save_preds = np.append(save_preds,save_ref, axis=0)
            
        save_preds= save_preds[1:,...]
        data = output_voxel_color_point_cloud(save_preds, os.path.join(save_obj_dir, '%d_pred.obj' % num))
        output_point_cloud(save_preds, os.path.join(save_obj_dir, '%d_pred.txt' % num))
        
         
        point_pred = data[:,:3]
        color_pred = data[:,3:]
            
        db_dir_h5 = os.path.join(db_dir, cat_id, obj, "models/voxel.h5")            
        model = h5py.File(db_dir_h5, 'r')
        point_gt = np.array(model['voxel_pt'][:])
        point_gt = point_gt[:int(point_gt.shape[0]/2),:]
        color_gt = np.array(model['voxel_val'][:])
        color_gt = color_gt[:int(color_gt.shape[0]/2),:]   
    
        iou ,psnr = metric(point_gt, point_pred, color_gt, color_pred)
        obj_iou.append(iou)
        obj_psnr.append(psnr)
            
#         print("%d Category %s obj %s num %d  IOU : %f, PSNR : %f" %(i,cat_id,obj,num, iou, psnr))
        log_string("%d Category %s obj %s num %d  IOU : %f, PSNR : %f" %(i,cat_id,obj,num, iou, psnr))
        img = batch_data['img'][0,...]
        plt.imsave(os.path.join(save_obj_dir, '%d.jpg' % num),img)
        pre_id = cat_id
        
        
    TEST_DATASET.shutdown()
    np.save(os.path.join(save_dir,pre_id,'result_iou'+ pre_id), result_iou)
    np.save(os.path.join(save_dir,pre_id,'result_psnr' +pre_id), result_psnr)



if __name__ == "__main__":


    # 1. create all categories / some of the categories:
    create()
