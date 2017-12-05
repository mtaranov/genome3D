import sys
import pickle
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
sys.path.append('/users/mtaranov/genome3D/')
from models_3d import Genome3D_RandomForest, Genome3D_SVM_RBF
from metrics import ClassificationResult
from utils import normalize_features_sampl_by_f, concat_motifs, get_features, get_labels, subsample_data, normalize_features

contacts='pe'
path='/users/mtaranov/datasets_3d/by_chr_dist_matched_'+contacts+'/'
day='d0'
thres='10'

X_train = np.load(path+'motifs/'+day+'_thres_'+thres+'_train_'+contacts+'_out_btw_nodes_3_0.0001/mat.npy')
X_test = np.load(path+'motifs/'+day+'_thres_'+thres+'_test_'+contacts+'_out_btw_nodes_3_0.0001/mat.npy')
X_valid = np.load(path+'motifs/'+day+'_thres_'+thres+'_valid_'+contacts+'_out_btw_nodes_3_0.0001/mat.npy')

y_train = get_labels(path+day+'_y_train_thres_'+thres+'.npy')
y_test = get_labels(path+day+'_y_test_thres_'+thres+'.npy')
y_valid = get_labels(path+day+'_y_valid_thres_'+thres+'.npy')

X_train_normalized, X_valid_normalized, X_test_normalized = normalize_features_sampl_by_f(X_train, X_valid, X_test)

X_train_valid = np.concatenate((X_train_normalized, X_valid_normalized), axis=0)
y_train_valid = np.concatenate((y_train, y_valid), axis=0)

# test_fold to 0 for all samples that are part of the validation set, and to -1 for all other samples.
valid_index=[-1 for i in range(X_train_normalized.shape[0])]+[0 for i in range(X_valid_normalized.shape[0])]

param_grid = {'gamma': [1e-3, 1e-4, 0.005, 0.05, 0.5],'C': [1, 10, 100]}
best_param={}
svm = Genome3D_SVM_RBF(best_param)
best_param = svm.train_cross_val(X_train_valid[:,:], [i for i in y_train_valid[:,0]], valid_index, param_grid)

with open('/users/mtaranov/genome3D/examples/from_sequence_features/edge_features/best_param_svm_rbf.pkl', 'wb') as f:
        pickle.dump(best_param, f, pickle.HIGHEST_PROTOCOL)

print 'svm_rbf_best_param: ', best_param
