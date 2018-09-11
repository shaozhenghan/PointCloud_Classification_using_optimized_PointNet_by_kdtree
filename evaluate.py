import tensorflow as tf
import numpy as np
import argparse
import importlib
import time
import os
import scipy.misc
import sys
from sklearn.decomposition import PCA
from scipy import spatial
import data_robust_test

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls_basic', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
# parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
# NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 5

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def scale_features(data, ax):
    # return (data - np.mean(data, axis=ax)) / np.std(data, axis=ax)
    # return (data - np.mean(data, axis=ax))
    pca = PCA(n_components=2, copy=False)
    data[:, 0:2] = pca.fit_transform(data[:, 0:2])
    data[:, :-1] = (data[:, :-1] - np.mean(data[:, :-1], axis=ax)) / np.array([11.344, 2.2207, 1.4886])
    data[:, -1] = data[:, -1] / np.array([0.99])
    return data

def evaluate():
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        loss = MODEL.get_loss(pred, labels_pl, end_points)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}

    eval_one_epoch(sess, ops)



def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]  

    path = "/media/shao/TOSHIBA EXT/data_object_velodyne/Daten_txt_CNN/test"
    filelist = os.listdir(path)
    np.random.shuffle(filelist)
    num_batches = len(filelist) // BATCH_SIZE
    for f in filelist:
        data_with_label = np.loadtxt(path + "/" + f)

        ################################### for robust test ############################
        # data_with_label = data_robust_test.getOccludedCloud(data_with_label, 90.0)
        # data_with_label = data_robust_test.getSparseCloud(data_with_label, 10.0)
        data_with_label = data_robust_test.addNoise(data_with_label, 0.09)
        print('data_with_label.shape for robust test', data_with_label.shape) # debug
        ################################################################################

        current_data = data_with_label[:, :-1] 
        current_data = scale_features(current_data, 0)  # don't forget the feature scaling!
        np.random.shuffle(current_data)
        # kd-tree
        tree = spatial.KDTree(current_data[:, :-1])
        length = len(current_data)/2
        current_data_ = []
        for i in range(length):
            if len(current_data) < 10:
                num_knn = len(current_data)
            else:
                num_knn = 10
            _, pidx = tree.query(current_data[i, :-1], num_knn)
            for pi in pidx:
                current_data_.append(current_data[pi])
        current_data_ = np.array(current_data_)
        current_data_ = current_data_[np.newaxis, ...]
        # print('current_data.shape', current_data.shape)
        current_label = data_with_label[1, -1]
        current_label = current_label.astype(np.int32)
        current_label = current_label[np.newaxis, ...]
        # print('current_label.shape', current_label.shape)
        # print('current_label', current_label)          

        feed_dict = {ops['pointclouds_pl']: current_data_,
                        ops['labels_pl']: current_label,
                        ops['is_training_pl']: is_training}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
        # print('eval_pred_val', pred_val) # debug
        pred_val = np.argmax(pred_val, 1)
        # print('eval_pred_val', pred_val) # debug
        correct = np.sum(pred_val == current_label)
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += (loss_val*BATCH_SIZE)
        total_seen_class[current_label[0]] += 1
        total_correct_class[current_label[0]] += (pred_val == current_label)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    

if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()
