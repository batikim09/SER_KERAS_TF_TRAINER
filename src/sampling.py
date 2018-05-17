import numpy as np
from imblearn.combine import SMOTEENN
import argparse
import h5py
import sys

def resample(X, Y, nb_class):
    print("original shape: ", X.shape)
    labels = Y.astype(int)
    counts = np.bincount(labels)
    
    if len(counts) != nb_class:
        print("there is no samples to interpolate! skip this fold.")
        return X, Y
        
    class_dist = counts / float(sum(counts))
    print("original dist: ", class_dist)
    
    org_shape = X.shape
    sampler = SMOTEENN(random_state=0)
    flattend_X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3] * X.shape[4]))
    X_resampled, Y_resampled = sampler.fit_sample(flattend_X, labels)
    X_resampled = X_resampled.reshape((X_resampled.shape[0], X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
    print("sampled shape: ", X_resampled.shape)
    
    Y_resampled = Y_resampled.astype(int)
    counts = np.bincount(Y_resampled)
    class_dist = counts / float(sum(counts))
    print("after SMOTEENN dist: ", class_dist)
    return X_resampled, Y_resampled

if __name__== "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-dt", "--data", dest= 'data', type=str, help="data")
    parser.add_argument("-idx", "--idx", dest= 'class_idx', type=int, help="class idx", default = 4)
    parser.add_argument("-nc", "--nb_class", dest= 'nb_class', type=int, help="the number of classes", default = 3)

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    with h5py.File(args.data,'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = hf.get('feat')
        train_csv = np.array(data)
        data = hf.get('label')
        train_lab = np.array(data)
        print('Shape of the array feat: ', train_csv.shape)
        print('Shape of the array lab: ', train_lab.shape)
        start_indice = np.array(hf.get('start_indice'))
        end_indice = np.array(hf.get('end_indice'))
        print('Shape of the indice for start: ', start_indice.shape)

    for i in range(0, len(start_indice)):
        start_idx = int(start_indice[i])
        end_idx = int(end_indice[i])
        print("start: ", start_idx)
        print("end: ", end_idx)
        X = train_csv[start_idx:end_idx, :]
        Y = train_lab[start_idx:end_idx, args.class_idx]

        resample(X, Y, args.nb_class)

        #TODO write into h5
