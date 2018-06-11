from __future__ import print_function
import numpy as np
np.random.seed(1337) 
import tensorflow as tf
tf.set_random_seed(2016)

from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Activation, TimeDistributed
from keras.layers.merge import Concatenate
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils
from keras import backend as K
from keras.layers.core import Permute
from keras.layers import Input, Conv3D,Conv2D,Conv1D, MaxPooling3D,MaxPooling2D, MaxPooling1D, GlobalAveragePooling1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger,TensorBoard
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import keras.utils.io_utils
import argparse
import h5py
import sys
from sklearn.metrics import f1_score,recall_score,confusion_matrix
from elm import ELM
from evaluation import *
from high_level import *
from highway import Highway 
from conv2d_highway import Conv2DHighway
from conv1d_highway import Conv1DHighway
from conv3d_highway import Conv3DHighway
from keras.models import load_model
from custom_cost import *
from sampling import *

def compile_model_with_custom_cost(model, multiTasks, dictForCost, dictForEval, Y_train, optimizer):
    for task, nb_classes, idx in multiTasks:
        if dictForCost[task] == 'weighted_categorical_crossentropy':
            init_w_categorical_crossentropy(dict_for_weighted_cost(Y_train[:,idx]))
            dictForCost[task] = w_categorical_crossentropy
        elif dictForCost[task] == 'categorical_focal_loss':
            #dictForCost[task] = CategoricalFocalLoss(nb_classes)
            init_categorical_focal_loss(nb_classes)
            dictForCost[task] = categorical_focal_loss
    model.compile(loss=dictForCost, optimizer=optimizer, metrics=dictForEval)
    return model

def utt_feature_ext(utt_feat_ext_layer, input):
    return utt_feat_ext_layer.predict(input)

def total_write_utt_feature(file, total_utt_features):
    f_handle = open(file,'w')
    
    np.savetxt(f_handle, total_utt_features)

    f_handle.close()

def compose_utt_feat(feature, multiTasks, labels, onehotvector = False):
    high_feat_label = np.zeros((feature.shape[0], feature.shape[1] + len(multiTasks)))
    print("high level feat shape: ", feature.shape)
    print("high level label shape: ", labels.shape, "multitasks: ", len(multiTasks), ": ", str(multiTasks))

    high_feat_label[:,0:feature.shape[1]] = feature

    id = 0
    for task, classes, idx in multiTasks:
        if onehotvector:
            high_feat_label[:,feature.shape[1] + id] = np.argmax(labels[:,idx],1)
        else:
            high_feat_label[:,feature.shape[1] + id] = labels[:,idx]
        id = id + 1

    return high_feat_label

def load_data_in_range(data_path, start, end, feat = 'feat', label = 'label'):
    X = np.array(keras.utils.io_utils.HDF5Matrix(data_path, feat, start, end))
    Y = np.array(keras.utils.io_utils.HDF5Matrix(data_path, label, start, end))
    return X, Y

def compose_idx(args_train_idx, args_test_idx, args_valid_idx, args_ignore_idx, args_adopt_idx, args_kf_idx):
    train_idx = []
    test_idx = []
    valid_idx = []
    ignore_idx = []
    adopt_idx = []
    kf_idx = []
    if args_train_idx:
        if ',' in args_train_idx:
            train_idx = [int(x) for x in args_train_idx.split(',')]
        elif ':' in args_train_idx:
            indice = args_train_idx.split(':')
            for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                train_idx.append(idx)
        else:
            train_idx = [int(x) for x in args_train_idx.split(',')]
    
    if args_test_idx:
        if ',' in args_test_idx:
            test_idx = [int(x) for x in args_test_idx.split(',')]
        elif ':' in args_test_idx:
            indice = args_test_idx.split(':')
            for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                test_idx.append(idx)
        else:
            test_idx = [int(x) for x in args_test_idx.split(',')]

    if args_ignore_idx:
        if ',' in args_ignore_idx:
            ignore_idx = [int(x) for x in args_ignore_idx.split(',')]
        elif ':' in args_ignore_idx:
            indice = args_ignore_idx.split(':')
            for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                ignore_idx.append(idx)
        else:
            ignore_idx = [int(x) for x in args_ignore_idx.split(',')]

    if args_valid_idx:
        if ',' in args_valid_idx:
            valid_idx = [int(x) for x in args_valid_idx.split(',')]
        elif ':' in args_valid_idx:
            indice = args_valid_idx.split(':')
            for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                valid_idx.append(idx)
        else:
            valid_idx = [int(x) for x in args_valid_idx.split(',')]
    if args_adopt_idx:
            if ',' in args_adopt_idx:
                adopt_idx = [int(x) for x in args_adopt_idx.split(',')]
            elif ':' in args_adopt_idx:
                indice = args_adopt_idx.split(':')
                for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                    adopt_idx.append(idx)
            else:
                adopt_idx = [int(x) for x in args_adopt_idx.split(',')]
    if args_kf_idx:
        kf_idx = args_kf_idx.split(",")
    kf_idx = set(kf_idx)
    
    return train_idx, test_idx, valid_idx, ignore_idx, adopt_idx, kf_idx

def evaluate_dataset(evaluation_set, X_train, X_valid, X_test, dictForLabelsTrain, dictForLabelsValid, dictForLabelsTest, multiTasks, class_weight_dict, reg, unweighted):
    test_scores = []
    for data in evaluation_set:
        if data == 'test':
            test_scores.append(predict_evaluate(X_test, dictForLabelsTest, multiTasks, class_weight_dict, reg, unweighted, data))
        elif data == 'valid':
            test_scores.append(predict_evaluate(X_valid, dictForLabelsValid, multiTasks, class_weight_dict, reg, unweighted, data))
        elif data == 'train':
            test_scores.append(predict_evaluate(X_train, dictForLabelsTrain, multiTasks, class_weight_dict, reg, unweighted, data))
    return test_scores

def temporal_evaluate_dataset(evaluation_set, model, X_train, X_test, X_valid, dictForLabelsTrain, dictForLabelsTest, dictForLabelsValid, multiTasks, elm_model_path, elm_m_task, unweighted, stl, elm_hidden, post_elm):
    test_scores = []
    for data in evaluation_set:
        if data == 'test':
            test_scores.append(predict_temporal_evaluate(model, X_train, X_test, X_valid, dictForLabelsTrain, dictForLabelsTest, dictForLabelsValid, multiTasks, elm_model_path, elm_m_task, unweighted, stl, elm_hidden, post_elm, data))
        elif data == 'valid':
            test_scores.append(predict_temporal_evaluate(model, X_train, X_valid, X_valid, dictForLabelsTrain, dictForLabelsValid, dictForLabelsValid, multiTasks, elm_model_path, elm_m_task, unweighted, stl, elm_hidden, post_elm, data))
        elif data == 'train':
            test_scores.append(predict_temporal_evaluate(model, X_train, X_train, X_valid, dictForLabelsTrain, dictForLabelsTrain, dictForLabelsValid, multiTasks, elm_model_path, elm_m_task, unweighted, stl, elm_hidden, post_elm, data))
    return test_scores
        
        
def predict_evaluate(X, Y, multiTasks, class_weight_dict, reg, unweighted, dataset):
    predictions = model.predict([X])
    if reg:
        test_scores = regression_task(predictions, Y, multiTasks, dataset)
    elif unweighted:
        test_scores = unweighted_recall_task(predictions, Y, multiTasks, class_weight_dict, dataset)
    else:
        test_scores = model.evaluate([X], Y, verbose=0)
    return test_scores

def predict_temporal_evaluate(model, X_train, X_test, X_valid, dictForLabelsTrain, dictForLabelsTest, dictForLabelsValid, multiTasks, elm_model_path, elm_m_task, unweighted = True, stl = False, elm_hidden = 40, post_elm = False, dataset = ''):
    if post_elm:   
        test_scores = elm_predict(model, X_train, X_test, X_valid, multiTasks, unweighted, stl,  dictForLabelsTrain, dictForLabelsTest, dictForLabelsValid, elm_hidden, elm_m_task, elm_save_path = elm_model_path, dataset = dataset)
    else:
        test_scores = frame_level_evaluation(model, X_test, dictForLabelsTemporalTest, multiTasks, stl, reg, dataset)
    return test_scores

def train_adopt_evaluate(model, multiTasks, X_train, X_test, X_valid, X_adopt, Y_train, Y_test, Y_valid, Y_adopt, max_t_steps, callbacks, elm_hidden, elm_m_task, utt_level = False, stl = False, unweighted = True, post_elm = True, model_save_path = './model/model', evaluation_set = [], r_valid = 0.0, epochs = 10, batch_size = 128, class_weights = None, reg = False):
    
    if reg:
        sample_weights = None
    else:
        sample_weights, class_weight_dict = generate_sample_weight(multiTasks, Y_train, class_weights)
        print("sample weight:, ", sample_weights)
        print("class weight:, ", class_weight_dict)
    
    print("max_t_steps: ", max_t_steps)
    
    main_model_path = model_save_path + ".h5"
    elm_model_path = model_save_path 

    dictForLabelsTemporalTest, dictForLabelsTemporalValid, dictForLabelsTemporalTrain, dictForLabelsTest, dictForLabelsValid, dictForLabelsTrain = generate_temporal_labels(multiTasks, Y_train, Y_test, Y_valid, max_t_steps)
    
    if len(X_adopt) != 0:
        dictForLabelsTemporalAdopt = generate_labels(multiTasks, Y_adopt, max_t_steps, True)
        dictForLabelsAdopt = generate_labels(multiTasks, Y_adopt, max_t_steps, False)
    
    print("Train shape: ", X_train.shape)
    print("batch_size: ", batch_size)
    print("epochs: ", epochs)
    print("r_valid: ", r_valid)
    
    if utt_level:
        if r_valid == 0.0:
            model.fit(X_train, dictForLabelsTrain, batch_size = batch_size, nb_epoch=epochs, validation_data=(X_valid, dictForLabelsValid), callbacks=callbacks, sample_weight = sample_weights)
        else:
            model.fit(X_train, dictForLabelsTrain, batch_size = batch_size, nb_epoch=epochs, validation_split = r_valid, callbacks=callbacks, sample_weight = sample_weights)
        
        if len(X_adopt) != 0:
            model.fit(X_adopt, dictForLabelsAdopt, batch_size = batch_size, nb_epoch=epochs, validation_split = r_valid, callbacks=callbacks, sample_weight = sample_weights)
        
        test_scores = evaluate_dataset(evaluation_set, X_train, X_valid, X_test, dictForLabelsTrain, dictForLabelsValid, dictForLabelsTest, multiTasks, class_weight_dict, reg, unweighted)
    else:
        if r_valid == 0.0:
            model.fit(X_train, dictForLabelsTemporalTrain, batch_size = batch_size, nb_epoch=epochs, validation_data=(X_valid, dictForLabelsTemporalValid), callbacks=callbacks, sample_weight = sample_weights)
        else:
            model.fit(X_train, dictForLabelsTemporalTrain, batch_size = batch_size, nb_epoch=epochs, validation_split = r_valid, callbacks=callbacks, sample_weight = sample_weights)
        
        if len(X_adopt) != 0:
            model.fit(X_adopt, dictForLabelsAdopt, batch_size = batch_size, nb_epoch=epochs, validation_split = r_valid, callbacks=callbacks, sample_weight = sample_weights)
            
            test_score = predict_temporal_evaluate(model, X_adopt, X_test, [], dictForLabelsAdopt, dictForLabelsTest, dictForLabelsValid, multiTasks, elm_model_path, elm_m_task, unweighted, stl, elm_hidden, post_elm)
        else:
            test_scores = temporal_evaluate_dataset(evaluation_set, model, X_train, X_test, X_valid, dictForLabelsTrain, dictForLabelsTest, dictForLabelsValid, multiTasks, elm_model_path, elm_m_task, unweighted, stl, elm_hidden, post_elm)

    if model_save_path != '':
        model.save(main_model_path)

    return test_scores
    

def evaluate(model, multiTasks, X_test, Y_test, max_t_steps, utt_level = True, unweighted = True, stl = True):
    dictForLabelsTemporalTest = generate_labels(multiTasks, Y_test, max_t_steps, temporal = True)
    dictForLabelsTest = generate_labels(multiTasks, Y_test, max_t_steps, temporal = False)
    
    if utt_level:
        predictions = model.predict([X_test])
        if reg:
            scores = regression_task(predictions, dictForLabelsTest, multiTasks, 'test')
        elif unweighted:
            scores = unweighted_recall_task(predictions, dictForLabelsTest, multiTasks, 'test')
        else:
            scores = model.evaluate([X_test], dictForLabelsTest, verbose=0)
    else:
        scores = frame_level_evaluation(model, X_test, dictForLabelsTemporalTest, multiTasks, stl, 'test')

    return [scores]

def write_result(scores, args, test_writer):
    for idx in range(len(scores)):
        score = scores[idx]
        print("evaluation set: ", args.evaluation_set[idx])
        result = str(score).replace('[','').replace(']','').replace(', ','\t')
        print(result)
        test_writer.write( args.evaluation_set[idx] + ":\t" + result + "\n")
        
if __name__== "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", dest= 'batch', type=int, help="batch size", default=128)
    parser.add_argument("-n_sub_b", "--nb_sub_batch", dest= 'nb_sub_batch', type=int, help="nb_sub_batch", default=10)
    parser.add_argument("-e", "--epoch", dest= 'epoch', type=int, help="number of epoch", default=50)
    parser.add_argument("-p", "--patience", dest= 'patience', type=int, help="patience size", default=5)
    parser.add_argument("-d", "--dropout", dest= 'dropout', type=float, help="dropout", default=0.0)
    parser.add_argument("-lr", "--learing_rate", dest= 'learingRate', type=float, help="learingRate", default=0.001)
    parser.add_argument("-l2", "--l2reg", dest= 'l2reg', type=float, help="l2reg", default=0.01)

    parser.add_argument("-cw", "--class_w", dest= 'class_w', type=str, help="class weights (e.g. arousal:0.5:1.0:0.5,valence:0.5:1.0:0.1)")
    parser.add_argument("-eval", "--evaluation_set", dest= 'evaluation_set', type=str, help="evaluation: test, valid, train", default="test")
    
    #dnn
    parser.add_argument("-nn", "--node_size", dest= 'node_size', type=int, help="DNN node_size", default=128)
    parser.add_argument("-dnn_depth", "--dnn_depth", dest= 'dnn_depth', type=int, help="depth of convolutional layers", default = 3)
    parser.add_argument("-f_dnn_depth", "--f_dnn_depth", dest= 'f_dnn_depth', type=int, help="depth of feature dnn", default = 2)
    parser.add_argument("-p_dnn_depth", "--p_dnn_depth", dest= 'p_dnn_depth', type=int, help="depth of post dnn", default = 2)

    parser.add_argument("-frozen", "--frozen", dest= 'frozen', type=str, help="(0,1,2,3,4)")
    parser.add_argument("-unloaded", "--unloaded", dest= 'unloaded', type=str, help="(0,1,2,3,4)")


    #convolution
    parser.add_argument("-n_row", "--nb_row", dest= 'nb_row', type=str, help="length of row for 2d convolution", default="10,5")
    parser.add_argument("-n_col", "--nb_col", dest= 'nb_col', type=str, help="length of column for 2d convolution", default="40,20")
    parser.add_argument("-n_time", "--nb_time", dest= 'nb_time', type=str, help="nb_time for 3d convolution", default="40,20")
    parser.add_argument("-l_filter", "--len_filter", dest= 'len_filter', type=str, help="filter length for 1d convolution", default="100,80")
    parser.add_argument("-n_filter", "--nb_filter", dest= 'nb_filter', type=str, help="nb_filter", default="40,20")
    parser.add_argument("-stride", "--stride", dest= 'sub_sample', type=int, help="stride (how many segment a filter shifts for each time", default=1)
    parser.add_argument("-pool", "--pool", dest= 'l_pool', type=str, help="pool", default="2,2")

    parser.add_argument("-pool_t", "--pool_t", dest= 'pool_t', type=str, help="pool", default="2")
    parser.add_argument("-pool_r", "--pool_r", dest= 'pool_r', type=str, help="pool", default="2")
    parser.add_argument("-pool_c", "--pool_c", dest= 'pool_c', type=str, help="pool", default="2")

    #lstm
    parser.add_argument("-c_len", "--context_len", dest= 'context_len', type=int, help="context_len", default=5)
    parser.add_argument("-nb_sample", "--nb_total_sample", dest= 'nb_total_sample', type=int, help="nb_total_sample")
    parser.add_argument("-cs", "--cell_size", dest= 'cell_size', type=int, help="LSTM cell_size", default=256)
    parser.add_argument("-t_max", "--t_max", dest= 't_max', type=int, help="max length of time sequence")

    #elm
    parser.add_argument("-elm_hidden", "--elm_hidden", dest= 'elm_hidden', type=int, help="elm_hidden", default=50)
    parser.add_argument("-elm_m_task", "--elm_m_task", dest= 'elm_m_task', type=int, help="elm_m_task", default=-1)

    #cv 
    parser.add_argument("-dt", "--data", dest= 'data', type=str, help="data")
    parser.add_argument("-kf", "--k_fold", dest= 'k_fold', type=int, help="random split k_fold")
    parser.add_argument("-n_cc", "--n_cc", dest= 'cc', type=str, help="cc (0,1,2,3,4)")
    parser.add_argument("-test_idx", "--test_idx", dest= 'test_idx', type=str, help="(0,1,2,3,4)")
    parser.add_argument("-train_idx", "--train_idx", dest= 'train_idx', type=str, help="(0,1,2,3,4)")
    parser.add_argument("-valid_idx", "--valid_idx", dest= 'valid_idx', type=str, help="(0,1,2,3,4)")
    parser.add_argument("-ignore_idx", "--ignore_idx", dest= 'ignore_idx', type=str, help="(0,1,2,3,4)")
    parser.add_argument("-adopt_idx", "--adopt_idx", dest= 'adopt_idx', type=str, help="Use train_idx together. Train data is first used then, adopted to this data(0,1,2,3,4)")
    parser.add_argument("-kf_idx", "--kf_idx", dest= 'kf_idx', type=str, help="(0,1,2,3,4)")
    parser.add_argument("-r_valid", "--r_valid", dest= 'r_valid', type=float, help="validation data rate from training", default=0.0)
    parser.add_argument("-mt", "--multitasks", dest= 'multitasks', type=str, help="multi-tasks (name:classes:idx:(cost_function):(weight)", default = 'acted:2:0::,arousal:2:1::')
    parser.add_argument("-ot", "--output_file", dest= 'output_file', type=str, help="output.txt", default="./output.txt")
    parser.add_argument("-sm", "--save_model", dest= 'save_model', type=str, help="save model", default='./model/model')
    parser.add_argument("-lm", "--load_model", dest= 'load_model', type=str, help="load pre-trained model. Only works with idx. Use train_idx for further training (not adaptation)")

    parser.add_argument("-log", "--log_file", dest= 'log_file', type=str, help="log file", default='./output/log.txt')
    parser.add_argument("-w_feat", "--w_feat", dest= 'w_feat', type=str, help="write feat file")
    parser.add_argument("-w_feat_layer", "--feature_ext_name", dest= 'feature_ext_name', type=str, help="write feat file")


    parser.add_argument("--conv", help="frame level convolutional network for 2d or 1d",
                        action="store_true")
    parser.add_argument("--conv_3d", help="frame level convolutional network for 3d",
                        action="store_true")
    parser.add_argument("--r_conv_3d", help="frame level convolutional network for 3d",
                        action="store_true")
    parser.add_argument("--conv_hw_3d", help="frame level convolutional network for 3d",
                        action="store_true")

    parser.add_argument("--r_conv", help="frame level residual convolutional network",
                        action="store_true")

    parser.add_argument("--f_dnn", help="frame level dnn requiring elm, otherwise it calculates frame-level performances",
                        action="store_true")
    parser.add_argument("--f_highway", help="frame level highway network requiring elm, otherwise it calculates frame-level performances",
                        action="store_true")
    parser.add_argument("--f_conv_highway", help="frame level 2d convolutional highway network requiring elm, otherwise it calculates frame-level performances",
                        action="store_true")
    parser.add_argument("--f_residual", help="frame level residual network requiring elm, otherwise it calculates frame-level performances",
                        action="store_true")

    parser.add_argument("--f_lstm", help="frame level lstm requiring elm, otherwise it calculates frame-level performances",
                        action="store_true")
    parser.add_argument("--u_lstm", help="utterance level lstm do not require elm",
                        action="store_true")
    parser.add_argument("--g_lstm", help="utterance level global average pooled lstm do not require elm",
                        action="store_true")
    parser.add_argument("--u_dnn", help="utterance level dnn after u_lstm do not require elm",
                        action="store_true")
    parser.add_argument("--u_hw", help="utterance level highway after u_lstm do not require elm",
                        action="store_true")
    parser.add_argument("--u_residual", help="utterance level residual after u_lstm do not require elm",
                        action="store_true")
    parser.add_argument("--g_pool", help="global pooling for temporal features",
                        action="store_true")

    parser.add_argument("--post_elm", help="elm for high-level feature modelling",
                        action="store_true")
    parser.add_argument("--headerless", help="headerless in feature file?",
                        action="store_true")
    parser.add_argument("--default", help="default training",
                        action="store_true")
    parser.add_argument("--log_append", help="append log or not",
                        action="store_true")
    parser.add_argument("--unweighted", help="unweighted evaluation",
                        action="store_true")
    parser.add_argument("--tb", help="tensorboard",
                        action="store_true")

    parser.add_argument("--reg", help="regression evaluation", action="store_true")
    parser.add_argument("--smote_enn", help="SMOTE ENN (STL only)", action="store_true")


    parser.add_argument("--decoding", help="decoding using a loaded model", action="store_true")
    parser.add_argument("--cw_predict",help="decoding using class weight", action="store_true")
    
    args = parser.parse_args()

    if len(sys.argv) == 1:

        parser.print_help()
        sys.exit(1)


    patience = args.patience
    batch_size = args.batch
    epochs = args.epoch
    nb_sub_batch = args.nb_sub_batch
    cell_size = args.cell_size
    node_size = args.node_size
    kfold = args.k_fold
    l2_reg = args.l2reg
    dropout = args.dropout
    learing_rate = args.learingRate
    tasks = args.multitasks 

    output_file = args.output_file
    log_file = args.log_file
    data_path = args.data
    
    args.evaluation_set = args.evaluation_set.split(',')
    # Conv
    nb_time = [] 
    for time in args.nb_time.split(','):
        nb_time.append(int(time))

    nb_row = [] 
    for row in args.nb_row.split(','):
        nb_row.append(int(row))

    nb_col = [] 
    for col in args.nb_col.split(','):
        nb_col.append(int(col))

    nb_filter = [] 
    for filter in args.nb_filter.split(','):
        nb_filter.append(int(filter))

    len_filter = []
    for length in args.len_filter.split(','):
        len_filter.append(int(length))

    l_pool = []
    for length in args.l_pool.split(','):
        l_pool.append(int(length))

    pool_t = []
    for length in args.pool_t.split(','):
        pool_t.append(int(length))


    pool_r = []
    for length in args.pool_r.split(','):
        pool_r.append(int(length))

    pool_c = []
    for length in args.pool_c.split(','):
        pool_c.append(int(length))

    if args.conv or args.r_conv:
        if len(nb_row) == len(nb_col) and len(nb_col) == len(nb_filter) and len(nb_filter) >= args.dnn_depth:
            print("correct setup for convolution")
        else:
            print("wrong setup for convolution, number of layers should match the number of setups for column, row, filters")
            print("depth: ", args.dnn_depth, "row: ", len(nb_row), "col: ", len(nb_col), "filter: ", len(nb_filter))
            exit()

    sub_sample = args.sub_sample

    save_model = args.save_model

    n_cc = []
    if args.cc:
        if ',' in args.cc:
            n_cc = args.cc.split(',')
        elif ':' in args.cc:
            indice = args.cc.split(':')
            for idx in range(int(indice[0]), int(indice[1]), +1):
                n_cc.append(idx)
        else:
            n_cc = args.cc.split(',')

        print('total cv: ', len(n_cc))


    r_valid = args.r_valid

    #utt level features
    total_utt_features = []

    #compose idx
    train_idx, test_idx, valid_idx, ignore_idx, adopt_idx, kf_idx = compose_idx(args.train_idx, args.test_idx, args.valid_idx, args.ignore_idx, args.adopt_idx, args.kf_idx)

    large_corpus_mode = False
    if args.nb_total_sample:
        print("very large corpus mode")
        large_corpus_mode = True
        train_csv, train_lab = load_data_in_range(data_path, 0, 1)
    else:
        with h5py.File(data_path,'r') as hf:
            print('List of arrays in this file: \n', hf.keys())
            data = hf.get('feat')
            train_csv = np.array(data)
            data = hf.get('label')
            train_lab = np.array(data).astype(int)
            print('Shape of the array feat: ', train_csv.shape)
            print('Shape of the array lab: ', train_lab.shape)
            if len(n_cc) or len(test_idx) > 0:
                start_indice = np.array(hf.get('start_indice'))
                end_indice = np.array(hf.get('end_indice'))
                print('Shape of the indice for start: ', start_indice.shape)

    input_type = "1d"

    #2d input
    if len(train_csv.shape) == 5:
        input_dim = train_csv.shape[4]
        context_len = train_csv.shape[3]
        if args.conv_3d or args.conv_hw_3d or args.r_conv_3d:
            input_type = "3d"
        else:    
            input_type = "2d"
    elif len(train_csv.shape) == 2:
        train_csv = np.reshape(train_csv, (train_csv.shape[0], 1, train_csv.shape[1]))
        input_dim = train_csv.shape[2]
    else:#1d input
        input_dim = train_csv.shape[2]

    max_t_steps = train_csv.shape[1]

    nameAndClasses = tasks.split(',')
    multiTasks = []
    dictForCost = {}
    dictForWeight = {}
    dictForEval = {}

    for task in nameAndClasses:
        params = task.split(':')

        if args.load_model:
            name = params[0] + "_un"
        else:
            name = params[0]

        name = params[0]

        nb_classes = int(params[1])
        idx = int(params[2])

        multiTasks.append((name, nb_classes, idx))
        if int(params[1]) == 1:    #regression problem
            dictForEval[name] = 'mean_squared_error'
        else:
            dictForEval[name] = 'accuracy'
        if params[3] != '':
            dictForCost[name] = params[3]
        else:
            dictForCost[name] = 'categorical_crossentropy'  
        if params[4] != '':
            dictForWeight[name] = float(params[4])
        else:
            dictForWeight[name] = 1.

    if len(multiTasks) > 1:
        stl = False
        print('MTL')       
    else:
        stl = True
        print('STL')

    #class weights

    if args.class_w:
        class_weights = {}
        if(args.class_w.startswith("auto") == True):
            print("class weights will be automatically set.")
        else:
            for weights in args.class_w.split(","):
                temp = weights.split(":")
                task = temp[0]
                w_s = np.zeros(len(temp) - 1)
                for idx in range(1, len(temp)):
                    w_s[idx - 1] = float(temp[idx])
                class_weights[task] = w_s

            print("class manual weights: ", class_weights)

    else:
        class_weights = None
        print("no class weights")

    #callbacks
    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss', patience=patience))
    if args.log_file:
        csv_logger = CSVLogger(log_file + ".csv", separator='\t')
        callbacks.append(csv_logger)

    if args.tb:
        callbacks.append(TensorBoard(log_dir = log_file + '.tb/', histogram_freq=2, write_graph=True, write_images=True))

    adam = Adam(lr=learing_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=0.5)

    print('Creating Model')

    #input
    if input_type == "3d":
        inputs = Input(shape=(1, args.t_max, context_len, input_dim),name='main_feat')
    elif input_type == "2d":
        inputs = Input(shape=(args.t_max, 1, context_len, input_dim),name='main_feat')
    elif input_type == "1d":
        inputs = Input(shape=(args.t_max, input_dim),name='main_feat')
    print('input type: ', input_type)    
    
    #batch normalisation
    batchNorm = BatchNormalization()

    #utterance level modelling
    if (args.f_dnn or args.f_lstm or args.f_highway or args.f_residual) and args.post_elm:
        utt_model = False
    else:
        utt_model = True

    #2d or 1d
    if input_type == "2d" or input_type == "1d":
        t_inputs = TimeDistributed(batchNorm)(inputs)

    if input_type == "3d":
        t_inputs = inputs
        if args.conv_3d:
            for d in range(args.dnn_depth):
                t_inputs = Conv3D(kernel_regularizer=regularizers.l2(l2_reg),filters=nb_filter[d],data_format="channels_first",
                                        kernel_size=(nb_time[d],nb_row[d],nb_col[d]),
                                        padding='same',
                                        activation='relu',
                                        strides=(sub_sample,sub_sample,sub_sample), name = 'conv-' + str(d))(t_inputs)
                t_inputs = MaxPooling3D(pool_size = (pool_t[d], pool_r[d], pool_c[d]), data_format="channels_first", name = 'pool-' + str(d))(t_inputs)
        elif args.conv_hw_3d:
            for d in range(args.dnn_depth):
                t_inputs = Conv3DHighway(kernel_regularizer=regularizers.l2(l2_reg),filters=nb_filter[d],data_format="channels_first",
                                        kernel_size=(nb_time[d],nb_row[d],nb_col[d]),
                                        padding='same',
                                        activation='relu',
                                        strides=(sub_sample,sub_sample,sub_sample), name = 'conv-hw-' + str(d))(t_inputs)
                t_inputs = MaxPooling3D(pool_size = (pool_t[d], pool_r[d], pool_c[d]), data_format="channels_first", name = 'pool-' + str(d))(t_inputs)
        elif args.r_conv_3d:
            t_inputs = Conv3D(kernel_regularizer=regularizers.l2(l2_reg),filters=nb_filter[0],data_format="channels_first",
                                        kernel_dim1=nb_time[0],
                                        kernel_dim2=nb_row[0],
                                        kernel_dim3=nb_col[0],
                                        padding='same',
                                        activation='relu',
                                        strides=(sub_sample,sub_sample,sub_sample), name = 'conv-' + str(0))(t_inputs)
            t_inputs = MaxPooling3D(pool_size = (pool_t[0], pool_r[0], pool_c[0]), data_format="channels_first", name = 'pool-' + str(0))(t_inputs)
            o_inputs = t_inputs
            for d in range(1, args.dnn_depth):
                t_inputs = Conv3D(kernel_regularizer=regularizers.l2(l2_reg),filters=nb_filter[d],data_format="channels_first",
                                        kernel_size=(nb_time[d],nb_row[d],nb_col[d]),
                                        padding='same',
                                        activation='relu',
                                        strides=(sub_sample,sub_sample,sub_sample), name = 'r-conv-' + str(d))(t_inputs)
                #residual merge and pooling at the end
                t_inputs = keras.layers.add([o_inputs, t_inputs], name="c_merge_" + str(d))
                t_inputs = MaxPooling3D(pool_size = (pool_t[d], pool_r[d], pool_c[d]), data_format="channels_first", name = 'pool-' + str(d))(t_inputs)
                o_inputs = t_inputs
        if args.u_lstm or args.f_lstm or args.f_dnn or args.f_residual or args.f_highway:
            t_inputs = Permute((2,1,3,4))(t_inputs)
            t_inputs = TimeDistributed(Flatten())(t_inputs)
        else:    
            t_inputs = Flatten()(t_inputs)

    elif input_type == "2d":
        if args.conv:
            for d in range(args.dnn_depth):
                t_inputs = TimeDistributed(Conv2D(kernel_regularizer=regularizers.l2(l2_reg),filters=nb_filter[d],data_format="channels_first",
                                    kernel_size=(nb_row[d],nb_col[d]),
                                    padding='same',
                                    activation='relu',
                                    strides=(sub_sample,sub_sample)), name = 'conv-' + str(d))(t_inputs)
                t_inputs = TimeDistributed(MaxPooling2D(pool_size = (pool_r[d], pool_c[d]), data_format="channels_first"), name = 'pool-' + str(d))(t_inputs)
        elif args.r_conv:

            t_inputs = TimeDistributed(Conv2D(kernel_regularizer=regularizers.l2(l2_reg),filters=nb_filter[0],data_format="channels_first",
                                    nb_row=nb_row[0],
                                    nb_col=nb_col[0],
                                    padding='same',
                                    activation='relu',
                                    strides=(sub_sample,sub_sample)), name = 'r-conv-' + str(0))(t_inputs)
            t_inputs = TimeDistributed(MaxPooling2D(pool_size = (pool_r[0], pool_c[0]), data_format="channels_first"), name = 'pool-' + str(0))(t_inputs)
            o_inputs = t_inputs
            for d in range(1, args.dnn_depth):
                t_inputs = TimeDistributed(Conv2D(kernel_regularizer=regularizers.l2(l2_reg),filters=nb_filter[d],data_format="channels_first",
                                    kernel_size=(nb_row[d],nb_col[d]),
                                    padding='same',
                                    activation='relu',
                                    strides=(sub_sample,sub_sample)), name = 'r-conv-' + str(d))(t_inputs)
                #residual merge and pooling at the end
                t_inputs = keras.layers.add([o_inputs, t_inputs], name="c_merge_" + str(d))
                t_inputs = TimeDistributed(MaxPooling2D(pool_size = (pool_r[d], pool_c[d]), data_format="channels_first"), name = 'pool-' + str(d))(t_inputs)
                o_inputs = t_inputs
        elif args.f_conv_highway:
            for d in range(args.dnn_depth):
                t_inputs = TimeDistributed(Conv2DHighway(kernel_regularizer=regularizers.l2(l2_reg),filters=nb_filter[d],data_format="channels_first",
                                    kernel_size=(nb_row[d],nb_col[d]),
                                    padding='same',
                                    activation='relu'), name = 'conv-hw-' + str(d))(t_inputs)
                t_inputs = TimeDistributed(MaxPooling2D(pool_size = (pool_r[d], pool_c[d]), data_format="channels_first"), name = 'pool-' + str(d))(t_inputs)

        #for all layers, flatten is necessary for 2d inputs
        t_inputs = TimeDistributed(Flatten())(t_inputs)
    elif input_type == "1d":#ID convolution
        for d in range(args.dnn_depth):
            if args.conv:
                t_inputs = Conv1D(filters=nb_filter[d],
                            filter_length=len_filter[d],
                            padding='valid',
                            activation='relu',
                            subsample_length=sub_sample)(t_inputs)
                t_inputs = MaxPooling1D(pool_length = l_pool[d])(t_inputs)
            elif args.f_conv_highway:
                t_inputs = Conv1DHighway(filters=nb_filter[d],
                            filter_length=len_filter[d],
                            padding='valid',
                            activation='relu',
                            subsample_length=sub_sample)(t_inputs)
                t_inputs = MaxPooling1D(pool_length = l_pool[d])(t_inputs)

    #for residual, highway dimension reduction
    if args.f_highway or args.f_residual:
        t_inputs = TimeDistributed(Dense(node_size, kernel_regularizer=regularizers.l2(l2_reg), activation = 'relu'), name = 'dim')(t_inputs)

    print("input shape for last feature before a time series of DNN", K.int_shape(t_inputs))
    max_t_steps = K.int_shape(t_inputs)[1]

    for d in range(args.f_dnn_depth):
        if args.f_dnn:
            t_inputs = TimeDistributed(Dense(node_size, kernel_regularizer=regularizers.l2(l2_reg), activation = 'relu'), name = 'fc-' + str(d))(t_inputs)
            if dropout > 0.0:
                t_inputs = TimeDistributed(Dropout(dropout))(t_inputs)
        elif args.f_highway:
            t_inputs = TimeDistributed(Highway(Dense(node_size, kernel_regularizer=regularizers.l2(l2_reg), activation = 'relu')), name = 'hw-' + str(d))(t_inputs)
            if dropout > 0.0:
                t_inputs = TimeDistributed(Dropout(dropout))(t_inputs)
        elif args.f_residual:
            d_1 = TimeDistributed(Dense(node_size, kernel_regularizer=regularizers.l2(l2_reg), activation = 'relu'), name = 'res-' + str(d)+ "-0")(t_inputs)
            d_2 = TimeDistributed(Dropout(dropout), name = 'd-' + str(d) + "-0")(d_1)
            t_inputs = keras.layers.add([d_2, t_inputs], name="c3d_merge_" + str(d))
    
    #frame level NNs but utterance level classifier (no elm)
    if (args.f_dnn or args.f_highway or args.f_residual) and utt_model:
        t_inputs = Flatten()(t_inputs)
        print("input shape for last feature before RNN", K.int_shape(t_inputs))
        
    #global pooling for utterance level lstm
    if args.g_pool and args.u_lstm:
        global_inputs = GlobalAveragePooling1D(name='global_pooling')(t_inputs)
    #LSTM
    if args.f_lstm:
        t_inputs = LSTM(cell_size, return_sequences = True, kernel_regularizer=regularizers.l2(l2_reg), recurrent_dropout=dropout, dropout=dropout, name = 'f-lstm-0')(t_inputs)
        #only if post_elm follows, return sequence
        if args.post_elm:
            t_inputs = LSTM(cell_size, return_sequences = True, kernel_regularizer=regularizers.l2(l2_reg), recurrent_dropout=dropout, dropout=dropout, name = 'f-lstm-1')(t_inputs)
        else:
            t_inputs = LSTM(cell_size, return_sequences = False, kernel_regularizer=regularizers.l2(l2_reg), recurrent_dropout=dropout, dropout=dropout, name = 'f-lstm-1')(t_inputs)
    if args.g_lstm:
        t_inputs = LSTM(cell_size, return_sequences = True, kernel_regularizer=regularizers.l2(l2_reg), recurrent_dropout=dropout, dropout=dropout, name = 'g-lstm-0')(t_inputs)
        t_inputs = LSTM(cell_size, return_sequences = True, kernel_regularizer=regularizers.l2(l2_reg), recurrent_dropout=dropout, dropout=dropout, name = 'g-lstm-1')(t_inputs)
        t_inputs = GlobalAveragePooling1D(name='global_pooling_lstm')(t_inputs)
    elif args.u_lstm:
        t_inputs = LSTM(cell_size, return_sequences = True, kernel_regularizer=regularizers.l2(l2_reg), recurrent_dropout=dropout, dropout=dropout, name = 'u-lstm-0')(t_inputs)
        t_inputs = LSTM(cell_size, return_sequences = False, kernel_regularizer=regularizers.l2(l2_reg), recurrent_dropout=dropout, dropout=dropout, name = 'u-lstm-1')(t_inputs)

        if args.g_pool:
            t_inputs = Concatenate(name= "merged_g_pool_lstm")([global_inputs, t_inputs])

    if args.u_hw or args.u_residual:
        t_inputs = Dense(node_size, kernel_regularizer=regularizers.l2(l2_reg), activation = 'relu', name = 'dim')(t_inputs)

    #post dnn for u-lstm or 3d convolution
    if utt_model:
        for d in range(args.p_dnn_depth):
            if args.u_dnn:
                layer_name = 'u-fc-' + str(d)
                t_inputs = Dense(node_size, kernel_regularizer=regularizers.l2(l2_reg), activation = 'relu', name = layer_name)(t_inputs)
                if dropout > 0.0:
                    t_inputs = Dropout(dropout)(t_inputs)
            elif args.u_hw:
                layer_name = 'u-hw-' + str(d)
                t_inputs = Highway(Dense(node_size, kernel_regularizer=regularizers.l2(l2_reg), activation = 'relu'), name = layer_name)(t_inputs)
                if dropout > 0.0:
                    t_inputs = Dropout(dropout)(t_inputs)
            elif args.u_residual:
                layer_name = 'u_merge_' + str(d)
                d_1 = Dense(node_size, kernel_regularizer=regularizers.l2(l2_reg), activation = 'relu', name = 'u_res-' + str(d)+ "-0")(t_inputs)
                d_2 = Dropout(dropout, name = 'd-' + str(d) + "-0")(d_1)
                t_inputs = keras.layers.add([d_2, t_inputs], name=layer_name)
    else:
        for d in range(args.p_dnn_depth):
            if args.u_dnn:
                layer_name = 'u-fc-' + str(d)
                t_inputs = TimeDistributed(Dense(node_size, kernel_regularizer=regularizers.l2(l2_reg), activation = 'relu'), name = layer_name)(t_inputs)
                if dropout > 0.0:
                    t_inputs = TimeDistributed(Dropout(dropout))(t_inputs)
            elif args.u_hw:
                layer_name = 'u-hw-' + str(d)
                t_inputs = TimeDistributed(Highway(Dense(node_size,kernel_regularizer=regularizers.l2(l2_reg), activation = 'relu')), name = layer_name)(t_inputs)
                if dropout > 0.0:
                    t_inputs = TimeDistributed(Dropout(dropout))(t_inputs)
            elif args.u_residual:
                layer_name = 'u_merge_' + str(d)
                d_1 = TimeDistributed(Dense(node_size, kernel_regularizer=regularizers.l2(l2_reg), activation = 'relu'), name = 'u_res-' + str(d) + "-0")(t_inputs)
                d_2 = TimeDistributed(Dropout(dropout), name = 'd-' + str(d) + "-0")(d_1)
                t_inputs = keras.layers.add([d_2, t_inputs], name=layer_name)

    predictions = []
    for task, classes, idx in multiTasks:
        #frame level modelling using high-level feature ELM
        if utt_model == False:
            if classes == 1: #regresssion problem
                predictions.append(TimeDistributed(Dense(classes), name=task)(t_inputs))
            else:
                predictions.append(TimeDistributed(Dense(classes, activation='softmax'),name=task)(t_inputs))
        else:
            #utterance level
            if classes == 1: #regresssion problem
                predictions.append(Dense(classes, name=task)(t_inputs))
            else:
                predictions.append(Dense(classes, activation='softmax',name=task)(t_inputs))

    model = Model(inputs=inputs, outputs=predictions)

    if args.load_model:

        print("Pre-trained model:", args.load_model)
        
        premodel = load_model(args.load_model, custom_objects={'Conv3DHighway': Conv3DHighway, 'Conv2DHighway': Conv2DHighway, 'Conv1DHighway': Conv1DHighway, 'Highway': Highway, 'w_categorical_crossentropy': w_categorical_crossentropy, 'categorical_focal_loss': categorical_focal_loss})
        
        premodel.summary()
        premodel.save_weights('./temp.w.h5')

        #freezing and unloading
        frozen_layer_list = []
        if args.frozen:
            if ',' in args.frozen:
                frozen_layer_list = args.frozen.split(',')
            elif ':' in args.frozen:
                indice = args.frozen.split(':')
                for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                    frozen_layer_list.append(idx)
            else:
                frozen_layer_list = args.frozen.split(",")

        unload_layer_list = []
        if args.unloaded:
            if ',' in args.unloaded:
                unload_layer_list = args.unloaded.split(',')
            elif ':' in args.unloaded:
                indice = args.unloaded.split(':')
                for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                    unload_layer_list.append(idx)
            else:
                unload_layer_list = args.unloaded.split(",")

        #total number of layers
        n_layers = len(model.layers)
        print("total layers: ", n_layers)

        for idx in frozen_layer_list:
            print("layer: ", model.layers[int(idx)].name, " is frozen")
            model.layers[int(idx)].trainable = False    

        for idx in unload_layer_list:
            if idx >= n_layers - len(multiTasks):
                print("You can't unload output layers;keep the same tasks")
                continue

            print("unloaded layer: ", model.layers[int(idx)].name)
            model.layers[int(idx)].name =  model.layers[int(idx)].name + "_un"

        #in decoding mode, model configuration should not affect
        if args.decoding == False:
            print("loading weights........")
            model.load_weights('./temp.w.h5', by_name=True)
        else:#decoding mode
            model = premodel
            print("decoding mode, no update at all")
            for layer in model.layers[:n_layers]:
                layer.trainable = False
                print(str(layer) + " is frozen. The wights will not be updated")

        del premodel

    print("New model: ")
    model.summary()

    #utterance level modelling and feature extraction mode
    #total number of layers
    n_layers = len(model.layers)
    if utt_model and args.w_feat:
        if args.feature_ext_name:
            utt_feat_ext_layer = Model(inputs=model.input, outputs=model.get_layer(args.feature_ext_name).output)
        else:
            utt_feat_ext_layer = Model(inputs=model.input, outputs=model.layers[n_layers - len(multiTasks) - 1].output)

    test_writer = open(output_file, 'a')
    test_writer.write(str(args) + "\n")
    print(args)
    
    if len(test_idx) > 0 :

        test_indice = []
        valid_indice = []
        adopt_indice = []
        remove_indice = []

        for cid in ignore_idx:
            print("cross-validation ignore: ", cid)
            start_idx = start_indice[int(cid)]
            end_idx = end_indice[int(cid)]

            if start_idx == 0 and end_idx == 0:
                continue

            for idx in range(int(start_idx), int(end_idx), + 1):
                remove_indice.append(idx)

        for cid in test_idx:
            print("cross-validation test: ", cid)
            start_idx = start_indice[int(cid)]
            end_idx = end_indice[int(cid)]

            if start_idx == 0 and end_idx == 0:
                continue

            for idx in range(int(start_idx), int(end_idx), + 1):
                test_indice.append(idx)
                remove_indice.append(idx)

        for cid in valid_idx:
            print("cross-validation valid: ", cid)
            start_idx = start_indice[int(cid)]
            end_idx = end_indice[int(cid)]

            if start_idx == 0 and end_idx == 0:
                continue

            for idx in range(int(start_idx), int(end_idx), + 1):
                remove_indice.append(idx)
                valid_indice.append(idx)

        for cid in adopt_idx:
            print("cross-adoptation adopt: ", cid)
            start_idx = start_indice[int(cid)]
            end_idx = end_indice[int(cid)]

            if start_idx == 0 and end_idx == 0:
                continue

            for idx in range(int(start_idx), int(end_idx), + 1):
                remove_indice.append(idx)
                adopt_indice.append(idx)

        if len(train_idx):
            train_indice = []
            for cid in train_idx:
                print("cross-validation train: ", cid)
                start_idx = start_indice[cid]
                end_idx = end_indice[cid]

                if start_idx == 0 and end_idx == 0:
                    continue

                for idx in range(int(start_idx), int(end_idx), + 1):
                    train_indice.append(idx)

            X_train = train_csv[train_indice]  
            Y_train = train_lab[train_indice]
        else:
            X_train = np.delete(train_csv, remove_indice, axis=0)
            Y_train = np.delete(train_lab, remove_indice, axis=0)

        if args.smote_enn and len(multiTasks) == 1:
            print("Only single task learning is supported for SMOTE ENN")
            X_train, Y_temp = resample(X_train, Y_train[:, multiTasks[0][2]], multiTasks[0][1])
            Y_train = np.zeros((Y_temp.shape[0], Y_train.shape[1]))
            Y_train[:, multiTasks[0][2]] = Y_temp
            Y_train = Y_train.astype(int)

        #test set
        X_test = train_csv[test_indice]  
        Y_test = train_lab[test_indice]

        #adopt set
        X_adopt = train_csv[adopt_indice]  
        Y_adopt = train_lab[adopt_indice]

        #valid set
        if len(valid_indice) == 0:
            X_valid = X_test
            Y_valid = Y_test
        else:
            X_valid = train_csv[valid_indice]  
            Y_valid = train_lab[valid_indice]
            r_valid = 0.0

        print('train shape: %s, %s' % (str(X_train.shape), str(Y_train.shape)))
        print('test shape: %s, %s' % (str(X_test.shape), str(Y_test.shape)))

        #re-compiling using custom cost function depending on 
        model = compile_model_with_custom_cost(model, multiTasks, dictForCost, dictForEval, Y_train, adam)

        scores = train_adopt_evaluate(model, multiTasks, X_train, X_test, X_valid, X_adopt, Y_train, Y_test, Y_valid, Y_adopt, max_t_steps, callbacks, args.elm_hidden, args.elm_m_task, utt_level = utt_model, stl = stl, unweighted = args.unweighted, post_elm = args.post_elm, model_save_path = save_model, evaluation_set = args.evaluation_set, r_valid = r_valid, epochs = epochs, batch_size = batch_size, class_weights = class_weights, reg = args.reg)

        #write results
        write_result(scores, args, test_writer)

        if args.w_feat:
            if utt_model:
                test_feat = utt_feature_ext(utt_feat_ext_layer, X_test)
                train_feat = utt_feature_ext(utt_feat_ext_layer, X_train)
                if len(X_adopt) > 0:
                    adopt_feat = utt_feature_ext(utt_feat_ext_layer, X_adopt)
                    total_write_utt_feature(args.w_feat + '.adopt.csv', compose_utt_feat(adopt_feat, multiTasks, Y_adopt))
                total_write_utt_feature(args.w_feat + '.test.csv', compose_utt_feat(test_feat, multiTasks, Y_test))            
                total_write_utt_feature(args.w_feat + '.train.csv', compose_utt_feat(train_feat, multiTasks, Y_train))
            else:
                total_write_high_feature(args.w_feat + '.test.csv', total_high_pred_test)
                total_write_high_feature(args.w_feat + '.train.csv', total_high_pred_train)

    elif len(n_cc) > 0:
        for cid in n_cc:
            print("cross-validation: ", cid)
            start_idx = start_indice[cid]
            end_idx = end_indice[cid]

            if start_idx == 0 and end_idx == 0:
                continue

            X_test = train_csv[start_idx:end_idx, :]  
            Y_test = train_lab[start_idx:end_idx, :]
            indices = range(int(start_idx), int(end_idx), +1)
            X_train = np.delete(train_csv, indices, axis=0)
            Y_train = np.delete(train_lab, indices, axis=0)
            X_valid = X_test
            Y_valid = Y_test#appending looks too computational.

            model = compile_model_with_custom_cost(model, multiTasks, dictForCost, dictForEval, Y_train, adam)

            if len(callbacks) == 2:
                callbacks[1] = CSVLogger(log_file + '.' + str(cid) + '.csv', separator='\t')
            
            scores = train_adopt_evaluate(model, multiTasks, X_train, X_test, X_valid, [], Y_train, Y_test, Y_valid, [], max_t_steps, callbacks, args.elm_hidden, args.elm_m_task, utt_level = utt_model, stl = stl, unweighted = args.unweighted, post_elm = args.post_elm, model_save_path = save_model + '.' + str(cid), evaluation_set = args.evaluation_set, r_valid = r_valid, epochs = epochs, batch_size = batch_size, class_weights = class_weights, reg = args.reg)

            #write results
            write_result(scores, args, test_writer)

            if args.w_feat:
                if utt_model:
                    test_feat = utt_feature_ext(utt_feat_ext_layer, X_test)
                    train_feat = utt_feature_ext(utt_feat_ext_layer, X_train)
                    total_write_utt_feature(args.w_feat + '.test.' + str(cid) + '.csv', compose_utt_feat(test_feat, multiTasks, Y_test))
                    total_write_utt_feature(args.w_feat + '.train.'+ str(cid) + '.csv', compose_utt_feat(train_feat, multiTasks, Y_train))
                else:
                    total_write_high_feature(args.w_feat + '.test.' + str(cid) + '.csv', total_high_pred_test)
                    total_write_high_feature(args.w_feat + '.train.'+ str(cid) + '.csv', total_high_pred_train)



    elif large_corpus_mode:
        sub_batch_size = batch_size * nb_sub_batch
        n_sample_per_fold = int( args.nb_total_sample / kfold)

        for ev_fold in range(kfold):

            if len(kf_idx) == 0 or str(ev_fold) in kf_idx:
                start_idx = ev_fold * n_sample_per_fold
                end_idx = start_idx + n_sample_per_fold
                print("k fold ID", ev_fold)
                print('evaluation starting from ', start_idx, " to ", end_idx)

                X_test, Y_test = load_data_in_range(data_path, start_idx, end_idx)

                X_valid = X_test
                Y_valid = Y_test

                model = compile_model_with_custom_cost(model, multiTasks, dictForCost, dictForEval, Y_train, adam)

                for epoch in range(epochs):
                    print("Main epoch:", epoch)
                    for tr_fold in range(kfold):
                        #preparing data sets for training
                        if tr_fold == ev_fold:
                            continue
                        start_idx = tr_fold * n_sample_per_fold
                        end_idx = start_idx + n_sample_per_fold
                        print("training starting from ", start_idx, " to ", end_idx)
                        X_train, Y_train = load_data_in_range(data_path, start_idx, end_idx)
                        
                        scores = train_adopt_evaluate(model, multiTasks, X_train, X_test, X_valid, [], Y_train, Y_test, Y_valid, [], max_t_steps, callbacks, args.elm_hidden, args.elm_m_task, utt_level = utt_model, stl = stl, unweighted = args.unweighted, post_elm = args.post_elm, model_save_path = '', evaluation_set = [], r_valid = r_valid, epochs = epochs, batch_size = batch_size, class_weights = class_weights, reg = args.reg)

                    #evaluation
                    scores = evaluate(model, multiTasks, X_test, Y_test, max_t_steps, utt_level = utt_model, unweighted = args.unweighted, stl = stl)
                    #write results
                    write_result(scores, args, test_writer)

                #model reset
                model.compile(loss=dictForCost, optimizer=adam, metrics=dictForEval)

    else:
        n_sample_per_fold = int(len(train_csv) / kfold)
        for fold in range(kfold):

            if len(kf_idx) == 0 or str(fold) in kf_idx:
                start_idx = fold * n_sample_per_fold
                end_idx = start_idx + n_sample_per_fold
                print("k fold ID", fold)
                print('starting from ', start_idx, " to ", end_idx)

                X_test = train_csv[start_idx:end_idx, :]  
                Y_test = train_lab[start_idx:end_idx, :]
                indices = range(start_idx, end_idx, +1)

                X_train = np.delete(train_csv, indices, axis=0)
                Y_train = np.delete(train_lab, indices, axis=0)

                X_valid = X_test
                Y_valid = Y_test

                model = compile_model_with_custom_cost(model, multiTasks, dictForCost, dictForEval, Y_train, adam)

                if len(callbacks) == 2:
                    callbacks[1] = CSVLogger(log_file + '.' + str(fold) + '.csv', separator='\t')

                scores = train_adopt_evaluate(model, multiTasks, X_train, X_test, X_valid, [], Y_train, Y_test, Y_valid, [], max_t_steps, callbacks, args.elm_hidden, args.elm_m_task, utt_level = utt_model, stl = stl, unweighted = args.unweighted, post_elm = args.post_elm, model_save_path = save_model + '.' + str(fold), r_valid = r_valid, epochs = epochs, batch_size = batch_size, class_weights = class_weights, reg = args.reg)

                #write results
                write_result(scores, args, test_writer)

                if args.w_feat:
                    if utt_model:
                        test_feat = utt_feature_ext(utt_feat_ext_layer, X_test)
                        train_feat = utt_feature_ext(utt_feat_ext_layer, X_train)

            model.compile(loss=dictForCost, optimizer=adam, metrics=dictForEval)


        if args.w_feat:
            if utt_model:
                test_feat = utt_feature_ext(utt_feat_ext_layer, X_test)
                train_feat = utt_feature_ext(utt_feat_ext_layer, X_train)
                total_write_utt_feature(args.w_feat + '.test.csv', compose_utt_feat(test_feat, multiTasks, Y_test))
                total_write_utt_feature(args.w_feat + '.train.csv', compose_utt_feat(train_feat, multiTasks, Y_train))
            else:
                total_write_high_feature(args.w_feat + '.test.csv', total_high_pred_test)
                total_write_high_feature(args.w_feat + '.train.csv', total_high_pred_train)

    if args.reg:
        total_write_ccc(test_writer)
    else:
        total_write_collected_cm(test_writer)
        total_write_cm(test_writer)

    test_writer.close()