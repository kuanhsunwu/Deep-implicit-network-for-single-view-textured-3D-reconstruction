import numpy as np
import cv2
import random
import math
import os
import threading
import queue
import sys
import h5py
import copy

FETCH_BATCH_SIZE = 32
BATCH_SIZE = 32
HEIGHT = 192
WIDTH = 256
POINTCLOUDSIZE = 16384
OUTPUTPOINTS = 1024
REEBSIZE = 1024


def get_filelist(lst_dir, maxnverts, minsurbinvox, cats, cats_info, type):
    for cat in cats:
        cat_id = cats_info[cat]
    inputlistfile = os.path.join(lst_dir, cat_id + type + ".lst")
    with open(inputlistfile, 'r') as f:
        lines = f.read().splitlines()
        file_lst = [[cat_id, line.strip()] for line in lines]
    return file_lst

class Pt_sdf_img(threading.Thread):
    def __init__(self, FLAGS, listinfo=None, info=None, qsize=64, cats_limit=None, shuffle=True):
        super(Pt_sdf_img, self).__init__()
        self.queue = queue.Queue(qsize)
        self.stopped = False
        self.bno = 0
        self.listinfo = listinfo
        self.num_points = FLAGS.num_points
        self.gen_num_pt = FLAGS.num_sample_points
        self.batch_size = FLAGS.batch_size
        self.img_dir = info['rendered_dir']
        self.sdf_dir = info['sdf_dir']
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 60000
        self.data_num = len(self.listinfo)
        self.FLAGS = FLAGS
        self.shuffle = shuffle
        self.num_batches = int(self.data_num / self.FLAGS.batch_size)
        self.cats_limit, self.epoch_amount = self.set_cat_limit(cats_limit)
        self.data_order = list(range(len(listinfo)))
        self.order = self.data_order

    
    def set_cat_limit(self, cats_limit):
        epoch_amount = 0
        for cat, amount in cats_limit.items():
            cats_limit[cat] = min(self.FLAGS.cat_limit, amount)
            epoch_amount += cats_limit[cat]
        print("epoch_amount ", epoch_amount)
#         print("cats_limit ", cats_limit)
        return cats_limit, epoch_amount

    def get_img_dir(self, cat_id, obj):
        img_dir = os.path.join(self.img_dir, cat_id, obj,"h5")
        return img_dir, None


    def get_sdf_h5_filenm(self, cat_id, obj):
        return os.path.join(self.sdf_dir, cat_id, obj, "models/voxel.h5")

    def pc_normalize(self, pc, centroid=None):

        """ pc: NxC, return NxC """
        l = pc.shape[0]

        if centroid is None:
            centroid = np.mean(pc, axis=0)

        pc = pc - centroid
        # m = np.max(pc, axis=0)
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))

        pc = pc / m

        return pc, centroid, m

    def __len__(self):
        return self.epoch_amount

    def memory(self):
        """
        Get node total memory and memory usage
        """
        with open('/proc/meminfo', 'r') as mem:
            ret = {}
            tmp = 0
            for i in mem:
                sline = i.split()
                if str(sline[0]) == 'MemTotal:':
                    ret['total'] = int(sline[1])
                elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                    tmp += int(sline[1])
            ret['free'] = tmp
            ret['used'] = int(ret['total']) - int(ret['free'])
        return ret

    def getitem(self, index):
        cat_id, obj, num = self.listinfo[index]
        sdf_file = self.get_sdf_h5_filenm(cat_id, obj)
        voxel_pt, voxel_val = self.get_sdf_h5(sdf_file, cat_id, obj)
        img_dir, img_file_lst = self.get_img_dir(cat_id, obj)
        return voxel_pt, voxel_val, img_dir, img_file_lst, cat_id, obj, num

    def get_sdf_h5(self, sdf_h5_file, cat_id, obj):
        h5_f = h5py.File(sdf_h5_file, 'r')
        try:
            if ('voxel_pt' in h5_f.keys() and 'voxel_val' in h5_f.keys()):
                voxel_pt = h5_f['voxel_pt'][:]
                voxel_val = h5_f['voxel_val'][:]
             
            else:
                raise Exception(cat_id, obj, "no sdf and sample")
        finally:
            h5_f.close()
        return voxel_pt, voxel_val

    def get_img_old(self, img_dir, num, file_lst):
        params = np.loadtxt(img_dir + "/rendering_metadata.txt")
        img_file = os.path.join(img_dir, file_lst[num])
        # azimuth, elevation, in-plane rotation, distance, the field of view.
        param = params[num, :].astype(np.float32)
        cam_mat, cam_pos = self.camera_info(self.degree2rad(param))
        img_arr = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)[:,:,:3].astype(np.float32) / 255.
        return img_arr, cam_mat, cam_pos

    def get_img(self, img_dir, num):
        img_h5 = os.path.join(img_dir, "%02d.h5"%num)
        trans_mat = None
        with h5py.File(img_h5, 'r') as h5_f:
            if self.FLAGS.img_feat_onestream or self.FLAGS.img_feat_twostream:
                trans_mat = h5_f["trans_mat"][:]
            else:
                cam_mat, cam_pos = h5_f["cam_mat"][:].astype(np.float32), h5_f["cam_pos"][:].astype(np.float32)
            if self.FLAGS.alpha:
                img_arr = h5_f["img_arr"][:]
            else:
                img_arr = h5_f["img_arr"][:]


            return img_arr, trans_mat

    def degree2rad(self, params):
        params[0] = np.deg2rad(params[0] + 180.0)
        params[1] = np.deg2rad(params[1])
        params[2] = np.deg2rad(params[2])
        return params

    def unit(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def camera_info(self, param):
        az_mat = self.get_az(param[0])
        el_mat = self.get_el(param[1])
        inl_mat = self.get_inl(param[2])
        cam_mat = np.transpose(np.matmul(np.matmul(inl_mat, el_mat), az_mat))
        cam_pos = self.get_cam_pos(param)
        return cam_mat, cam_pos

    def get_cam_pos(self, param):
        camX = 0
        camY = 0
        camZ = param[3]
        cam_pos = np.array([camX, camY, camZ])
        return -1 * cam_pos

    def get_az(self, az):
        cos = np.cos(az)
        sin = np.sin(az)
        mat = np.asarray([cos, 0.0, sin, 0.0, 1.0, 0.0, -1.0*sin, 0.0, cos], dtype=np.float32)
        mat = np.reshape(mat, [3,3])
        return mat
    #
    def get_el(self, el):
        cos = np.cos(el)
        sin = np.sin(el)
        mat = np.asarray([1.0, 0.0, 0.0, 0.0, cos, -1.0*sin, 0.0, sin, cos], dtype=np.float32)
        mat = np.reshape(mat, [3,3])
        return mat
    #
    def get_inl(self, inl):
        cos = np.cos(inl)
        sin = np.sin(inl)
        # zeros = np.zeros_like(inl)
        # ones = np.ones_like(inl)
        mat = np.asarray([cos, -1.0*sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        mat = np.reshape(mat, [3,3])
        return mat

    def get_batch(self, index):
        if index + self.batch_size > self.epoch_amount:
            index = index + self.batch_size - self.epoch_amount
        batch_voxel_pt = np.zeros((self.batch_size, self.gen_num_pt, 3), dtype=float)
        batch_occupied_pt = np.zeros((self.batch_size, self.gen_num_pt, 3), dtype=float)
        batch_voxel_val = np.zeros((self.batch_size, self.gen_num_pt, 3), dtype=float)

        if self.FLAGS.alpha:
            batch_img = np.zeros((self.batch_size, self.FLAGS.img_h, self.FLAGS.img_w, 4), dtype=np.float32)
        else:
            batch_img = np.zeros((self.batch_size, self.FLAGS.img_h, self.FLAGS.img_w, 3), dtype=np.float32)
        batch_trans_mat = np.zeros((self.batch_size, 4, 3), dtype=np.float32)
        batch_cat_id = []
        batch_obj_nm = []
        batch_view_id = []
        cnt = 0
        for i in range(index, index + self.batch_size):
            single_obj = self.getitem(self.order[i])
            if single_obj == None:
                raise Exception("single mesh is None!")
            voxel_pt, voxel_val, img_dir, img_file_lst, cat_id, obj, num = single_obj
            img, trans_mat = self.get_img(img_dir, num)

            if self.gen_num_pt > voxel_pt.shape[0]:
                choice = np.random.randint(voxel_pt.shape[0], size=self.gen_num_pt)
            else:
                choice = np.asarray(random.sample(range(voxel_pt.shape[0]), self.gen_num_pt), dtype=np.int32)
                
            if self.gen_num_pt > (voxel_pt.shape[0]/2):
                choice_occupy = np.random.randint(int(voxel_pt.shape[0]/2), size=self.gen_num_pt)
            else:
                choice_occupy = np.asarray(random.sample(range(int(voxel_pt.shape[0]/2)), self.gen_num_pt), dtype=np.int32)    
                
            batch_voxel_pt[cnt, ...] = voxel_pt[choice, :]
            batch_voxel_val[cnt, ...] = voxel_val[choice, :]

            batch_occupied_pt[cnt, ...] = voxel_pt[choice_occupy, :]           
            batch_img[cnt, ...] = img
            batch_trans_mat[cnt, ...] = trans_mat
            batch_cat_id.append(cat_id)
            batch_obj_nm.append(obj)
            batch_view_id.append(num)
            cnt += 1
        batch_data = {}
        batch_data['voxel_pt'] = batch_voxel_pt
        batch_data['voxel_val'] = batch_voxel_val
        batch_data['occupied_pt'] = batch_occupied_pt
        batch_data['img'] = batch_img
        batch_data['trans_mat'] = batch_trans_mat
        batch_data['cat_id'] = batch_cat_id
        batch_data['obj_nm'] = batch_obj_nm
        batch_data['view_id'] = batch_view_id
        return batch_data

    def refill_data_order(self):
        temp_order = copy.deepcopy(self.data_order)
        cats_quota = {key: value for key, value in self.cats_limit.items()}
        np.random.shuffle(temp_order)
        pointer = 0
        epoch_order=[]
        while len(epoch_order) < self.epoch_amount:
            cat_id, _, _ = self.listinfo[temp_order[pointer]]
            if cats_quota[cat_id] > 0:
                epoch_order.append(temp_order[pointer])
                cats_quota[cat_id]-=1
            pointer+=1
        return epoch_order


    def work(self, epoch, index):
        if index == 0 and self.shuffle:
            self.order = self.refill_data_order()
            print("data order reordered!")
        return self.get_batch(index)

    def run(self):
        while (self.bno // (self.num_batches* self.batch_size)) < self.FLAGS.max_epoch and not self.stopped:
            self.queue.put(self.work(self.bno // (self.num_batches* self.batch_size),
                                     self.bno % (self.num_batches * self.batch_size)))
            self.bno += self.batch_size

    def fetch(self):
        if self.stopped:
            return None
        # else:
        #     print("queue length", self.queue.qsize())
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()

if __name__ == '__main__':

    sys.path.append('../preprocessing/')
    import create_file_lst as create

    data = Pt_sdf_img(res=256, expr=1.5,
                      listinfo=[["03001627", "ff3581996365bdddc3bd24f986301745"],
                                ["03001627", "ff3581996365bdddc3bd24f986301745"]],
                      info=create.get_all_info(), maxnverts=6000, maxntris=50000,
                      minsurbinvox=4096, num_points=2048, batch_size=2, normalize=False, norm_color=True)
    batch1 = data.get_batch(0)
    print(batch1.keys())
    print(batch1["verts"].shape)
    print(batch1["nverts"])
    print(batch1["tris"].shape)
    print(batch1["ntris"])
    print(batch1["surfacebinvoxpc"].shape)
    print(batch1["sdf"].shape)
    print(batch1["sdf_params"])
    print(batch1["img"].shape, batch1["img"][0, 64, 64, :])
    print(batch1["img_cam"])

    # (2048, 3)
    cloud1 = batch1["surfacebinvoxpc"][0, ...]
    trans1 = batch1["img_cam"][0, ...]
    az1 = float(trans1[0] + 180) * math.pi / 180.0
    el1 = float(trans1[1]) * math.pi / 180.0
    in1 = float(trans1[2]) * math.pi / 180.0
    transmatrix_az1 = [[math.cos(az1), 0, math.sin(az1)],
                       [0, 1, 0],
                       [-math.sin(az1), 0, math.cos(az1)]]
    transmatrix_az1 = np.asarray(transmatrix_az1).astype(np.float32)
    transmatrix_el1 = [[1, 0, 0],
                       [0, math.cos(el1), -math.sin(el1)],
                       [0, math.sin(el1), math.cos(el1)]]
    transmatrix_el1 = np.asarray(transmatrix_el1).astype(np.float32)
    transmatrix_in1 = [[math.cos(in1), -math.sin(in1), 0],
                       [math.sin(in1), math.cos(in1), 0],
                       [0, 0, 1]]
    transmatrix_in1 = np.asarray(transmatrix_in1).astype(np.float32)

    trans = np.matmul(np.matmul(transmatrix_in1, transmatrix_el1), transmatrix_az1)
    translate1 = np.tile(np.expand_dims(np.asarray([-trans1[2], 0, 0]).astype(np.float32), axis=0), (2048, 1))
    points = np.matmul(cloud1, trans.T)
    np.savetxt("ff_rotate.xyz", points)
    np.savetxt("ff.xyz", cloud1)
