
#A time series of fully-connected-neural networks(FCN) + Flatten + FCN
python ./src/trainer.py -dnn_depth 3 --f_dnn --u_dnn -r_valid 0.1 -b 128 -e 5 -p 2 -nn 128 -l2 0.001 -t_max 10 -dt ../SER_FEAT_EXT/h5db/ENT.MSPEC.2d.3cls.av.h5 -mt arousal:3:4::,valence:3:5:: -ot ./output/sanity.txt -sm ./model/sanity -log ./output/sanity --unweighted -test_idx 0,1,2,3,4,5,6,7,8,9 -valid_idx 10,11,12,13,14,15,16,17,18,19

#LSTM + FCN
python ./src/trainer.py -dnn_depth 2 --f_lstm --u_dnn -r_valid 0.1 -b 128 -e 5 -p 2 -nn 128 -cs 64 -l2 0.001 -t_max 10 -dt ../SER_FEAT_EXT/h5db/ENT.MSPEC.2d.3cls.av.h5 -mt arousal:3:4::,valence:3:5:: -ot ./output/sanity.txt -sm ./model/sanity -log ./output/sanity --unweighted -test_idx 0,1,2,3,4,5,6,7,8,9 -valid_idx 10,11,12,13,14,15,16,17,18,19

#2D-CNN + LSTM
python ./src/trainer.py -dnn_depth 2 --conv --u_lstm -r_valid 0.1 -b 128 -e 5 -p 2 -cs 64 -nn 128 -n_time 2,2 -n_row 2,2 -n_col 2,2 -n_filter 2,4 -pool_t 2,4 -pool_r 2,4 -pool_c 2,4 -l2 0.001 -t_max 10 -dt ../SER_FEAT_EXT/h5db/ENT.MSPEC.2d.3cls.av.h5 -mt arousal:3:4::,valence:3:5:: -ot ./output/sanity.txt -sm ./model/sanity -log ./output/sanity --unweighted -test_idx 0,1,2,3,4,5,6,7,8,9 -valid_idx 10,11,12,13,14,15,16,17,18,19

#2D-CNN + LSTM + FCN
python ./src/trainer.py -dnn_depth 2 --conv --u_lstm --u_dnn -r_valid 0.1 -b 128 -e 5 -p 2 -cs 64 -nn 128 -n_time 2,2 -n_row 2,2 -n_col 2,2 -n_filter 2,4 -pool_t 2,4 -pool_r 2,4 -pool_c 2,4 -l2 0.001 -t_max 10 -dt ../SER_FEAT_EXT/h5db/ENT.MSPEC.2d.3cls.av.h5 -mt arousal:3:4::,valence:3:5:: -ot ./output/sanity.txt -sm ./model/sanity -log ./output/sanity --unweighted -test_idx 0,1,2,3,4,5,6,7,8,9 -valid_idx 10,11,12,13,14,15,16,17,18,19

#2D-RESNET + LSTM + RESNET (pooling is not allowed.)
python ./src/trainer.py -dnn_depth 2 --r_conv --u_lstm --u_residual -r_valid 0.1 -b 128 -e 5 -p 2 -cs 64 -nn 128 -n_time 2,2 -n_row 2,2 -n_col 2,2 -n_filter 2,2 -pool_t 1,1 -pool_r 1,1 -pool_c 1,1 -l2 0.001 -t_max 10 -dt ../SER_FEAT_EXT/h5db/ENT.MSPEC.2d.3cls.av.h5 -mt arousal:3:4::,valence:3:5:: -ot ./output/sanity.txt -sm ./model/sanity -log ./output/sanity --unweighted -test_idx 0,1,2,3,4,5,6,7,8,9 -valid_idx 10,11,12,13,14,15,16,17,18,19

#2D-HIGHWAY + LSTM + HIGHWAY (pooling is not allowed.)
python ./src/trainer.py -dnn_depth 2 --f_conv_highway --u_lstm --u_hw -r_valid 0.1 -b 128 -e 5 -p 2 -cs 64 -nn 128 -n_time 2,2 -n_row 2,2 -n_col 2,2 -n_filter 2,2 -pool_t 1,1 -pool_r 1,1 -pool_c 1,1 -l2 0.001 -t_max 10 -dt ../SER_FEAT_EXT/h5db/ENT.MSPEC.2d.3cls.av.h5 -mt arousal:3:4::,valence:3:5:: -ot ./output/sanity.txt -sm ./model/sanity -log ./output/sanity --unweighted -test_idx 0,1,2,3,4,5,6,7,8,9 -valid_idx 10,11,12,13,14,15,16,17,18,19

#3D-CNN + FCN
python ./src/trainer.py -dnn_depth 3 --conv_3d --u_dnn -r_valid 0.1 -b 128 -e 5 -p 2 -cs 64 -nn 128 -n_time 2,2,2 -n_row 2,2,2 -n_col 2,2,2 -n_filter 2,2,2 -pool_t 1,2,2 -pool_r 1,2,2 -pool_c 1,2,2 -l2 0.001 -t_max 10 -dt ../SER_FEAT_EXT/h5db/ENT.MSPEC.3d.3cls.av.h5 -mt arousal:3:4::,valence:3:5:: -ot ./output/sanity.txt -sm ./model/sanity -log ./output/sanity --unweighted -test_idx 0,1,2,3,4,5,6,7,8,9 -valid_idx 10,11,12,13,14,15,16,17,18,19

#3D-RESNET + RESNET
python ./src/trainer.py -dnn_depth 3 --r_conv_3d --u_dnn -r_valid 0.1 -b 128 -e 5 -p 2 -cs 64 -nn 128 -n_time 2,2,2 -n_row 2,2,2 -n_col 2,2,2 -n_filter 2,2,2 -pool_t 1,2,2 -pool_r 1,2,2 -pool_c 1,2,2 -l2 0.001 -t_max 10 -dt ../SER_FEAT_EXT/h5db/ENT.MSPEC.3d.3cls.av.h5 -mt arousal:3:4::,valence:3:5:: -ot ./output/sanity.txt -sm ./model/sanity -log ./output/sanity --unweighted -test_idx 0,1,2,3,4,5,6,7,8,9 -valid_idx 10,11,12,13,14,15,16,17,18,19
