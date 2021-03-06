import sys
import pickle
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
sys.path.append('/users/mtaranov/genome3D/')
from models_3d import Genome3D_RandomForest
from metrics import ClassificationResult
from utils import normalize_features_sampl_by_f, concat_motifs, get_features, get_labels, subsample_data, normalize_features

contacts='pe'
path='/users/mtaranov/datasets_3d/by_chr_dist_matched_'+contacts+'/'
day='d0'
thres='10'

X_train_node1 = path+'motifs/'+day+'_thres_'+thres+'_train_'+contacts+'_out_node1_3_0.0001/mat.npy'
X_train_node2 = path+'motifs/'+day+'_thres_'+thres+'_train_'+contacts+'_out_node2_3_0.0001/mat.npy'
X_train_window = path+'motifs/'+day+'_thres_'+thres+'_train_'+contacts+'_out_btw_nodes_3_0.0001/mat.npy'


X_test_node1 = path+'motifs/'+day+'_thres_'+thres+'_test_'+contacts+'_out_node1_3_0.0001/mat.npy'
X_test_node2 = path+'motifs/'+day+'_thres_'+thres+'_test_'+contacts+'_out_node2_3_0.0001/mat.npy'
X_test_window = path+'motifs/'+day+'_thres_'+thres+'_test_'+contacts+'_out_btw_nodes_3_0.0001/mat.npy'

X_valid_node1 = path+'motifs/'+day+'_thres_'+thres+'_valid_'+contacts+'_out_node1_3_0.0001/mat.npy'
X_valid_node2 = path+'motifs/'+day+'_thres_'+thres+'_valid_'+contacts+'_out_node2_3_0.0001/mat.npy'
X_valid_window = path+'motifs/'+day+'_thres_'+thres+'_valid_'+contacts+'_out_btw_nodes_3_0.0001/mat.npy'

y_train = get_labels(path+day+'_y_train_thres_'+thres+'.npy')
y_test = get_labels(path+day+'_y_test_thres_'+thres+'.npy')
y_valid = get_labels(path+day+'_y_valid_thres_'+thres+'.npy')

X_train=concat_motifs([X_train_node1, X_train_node2])
X_test=concat_motifs([X_test_node1, X_test_node2])
X_valid=concat_motifs([X_valid_node1, X_valid_node2])

X_train_normalized_pairs, X_valid_normalized_pairs, X_test_normalized_pairs = normalize_features(X_train, X_valid, X_test)

X_train_normalized = X_train_normalized_pairs.reshape(X_train_normalized_pairs.shape[0],X_train_normalized_pairs.shape[2]*X_train_normalized_pairs.shape[3])
X_valid_normalized = X_valid_normalized_pairs.reshape(X_valid_normalized_pairs.shape[0],X_valid_normalized_pairs.shape[2]*X_valid_normalized_pairs.shape[3])
X_test_normalized = X_test_normalized_pairs.reshape(X_test_normalized_pairs.shape[0],X_test_normalized_pairs.shape[2]*X_test_normalized_pairs.shape[3])

X_train_valid = np.concatenate((X_train_normalized, X_valid_normalized), axis=0)
y_train_valid = np.concatenate((y_train, y_valid), axis=0)

# test_fold to 0 for all samples that are part of the validation set, and to -1 for all other samples.
valid_index=[-1 for i in range(X_train_normalized.shape[0])]+[0 for i in range(X_valid_normalized.shape[0])]

param_grid = {'n_estimators': [100, 200, 500], 'criterion': ['gini', 'entropy'], 'max_features': ['auto', 'log2' ], 'max_depth': [None, 2, 3, 5], 'min_samples_split': [2, 3, 4], 'min_samples_leaf': [1, 2, 3], 'min_impurity_split': [0.0 , 0.1, 0.2]}

best_param={}
rf = Genome3D_RandomForest(best_param)
best_param = rf.train_cross_val(X_train_valid[:,:], [i for i in y_train_valid[:,0]], valid_index, param_grid)

with open('/users/mtaranov/genome3D/examples/from_sequence_features/nodes_features/best_param_rf.pkl', 'wb') as f:
        pickle.dump(best_param, f, pickle.HIGHEST_PROTOCOL)

print 'rf_best_param: ', best_param
