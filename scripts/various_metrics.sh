#You can choose various metrics for optimisation. Do not confuse it with loss

#Optimising by using accuracy (default)
python ./src/trainer.py -d 0.7 -dnn_depth 2 -p_dnn_depth 1 --conv --u_lstm --u_dnn -r_valid 0.1 -b 256 -e 40 -p 10 -cs 128 -nn 128 -n_time 2,2 -n_row 80,400 -n_col 1,1 -n_filter 20,20 -pool_t 2,4 -pool_r 10,20 -pool_c 1,1 -l2 0.001 -t_max 5 -dt ../SER_FEAT_EXT/h5db/AMI.500.RAW.laugh.h5 -mt class:2:0::: -test_idx 0 -r_valid 0.2 -ot ./output/ami.500.raw.cnnlstmfcn.c128.f1.txt -sm ./model/ami.500.raw.cnnlstmfcn.c128.f1.0 -log ./output/ami.500.raw.cnnlstmfcn.c128.f1.0

#Optimising by using F1
python ./src/trainer.py -d 0.7 -dnn_depth 2 -p_dnn_depth 1 --conv --u_lstm --u_dnn -r_valid 0.1 -b 256 -e 40 -p 10 -cs 128 -nn 128 -n_time 2,2 -n_row 80,400 -n_col 1,1 -n_filter 20,20 -pool_t 2,4 -pool_r 10,20 -pool_c 1,1 -l2 0.001 -t_max 5 -dt ../SER_FEAT_EXT/h5db/AMI.500.RAW.laugh.h5 -mt class:2:0:::f1 -test_idx 0 -r_valid 0.2 -ot ./output/ami.500.raw.cnnlstmfcn.c128.f1.txt -sm ./model/ami.500.raw.cnnlstmfcn.c128.f1.0 -log ./output/ami.500.raw.cnnlstmfcn.c128.f1.0

#Optimising by using precision
python ./src/trainer.py -d 0.7 -dnn_depth 2 -p_dnn_depth 1 --conv --u_lstm --u_dnn -r_valid 0.1 -b 256 -e 40 -p 10 -cs 128 -nn 128 -n_time 2,2 -n_row 80,400 -n_col 1,1 -n_filter 20,20 -pool_t 2,4 -pool_r 10,20 -pool_c 1,1 -l2 0.001 -t_max 5 -dt ../SER_FEAT_EXT/h5db/AMI.500.RAW.laugh.h5 -mt class:2:0:::precision -test_idx 0 -r_valid 0.2 -ot ./output/ami.500.raw.cnnlstmfcn.c128.f1.txt -sm ./model/ami.500.raw.cnnlstmfcn.c128.f1.0 -log ./output/ami.500.raw.cnnlstmfcn.c128.f1.0