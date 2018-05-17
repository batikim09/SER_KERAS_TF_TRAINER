
#without balanced learning
python ./src/trainer.py -dnn_depth 3 --f_dnn --u_dnn -r_valid 0.1 -b 128 -e 5 -p 2 -nn 128 -l2 0.001 -t_max 10 -dt ../SER_FEAT_EXT/h5db/ENT.MSPEC.2d.3cls.av.h5 -mt arousal:3:4::,valence:3:5:: -ot ./output/sanity.txt -sm ./model/sanity -log ./output/sanity --unweighted -test_idx 0,1,2,3,4,5,6,7,8,9 -valid_idx 10,11,12,13,14,15,16,17,18,19

#cost weighted prediction
python ./src/trainer.py -cw auto -dnn_depth 3 --f_dnn --u_dnn -r_valid 0.1 -b 128 -e 5 -p 2 -nn 128 -l2 0.001 -t_max 10 -dt ../SER_FEAT_EXT/h5db/ENT.MSPEC.2d.3cls.av.h5 -mt arousal:3:4::,valence:3:5:: -ot ./output/sanity.txt -sm ./model/sanity -log ./output/sanity --unweighted -test_idx 0,1,2,3,4,5,6,7,8,9 -valid_idx 10,11,12,13,14,15,16,17,18,19

#cost weighted learning
python ./src/trainer.py -dnn_depth 3 --f_dnn --u_dnn -r_valid 0.1 -b 128 -e 5 -p 2 -nn 128 -l2 0.001 -t_max 10 -dt ../SER_FEAT_EXT/h5db/ENT.MSPEC.2d.3cls.av.h5 -mt arousal:3:4:weighted_categorical_crossentropy:,valence:3:5:weighted_categorical_crossentropy: -ot ./output/sanity.txt -sm ./model/sanity -log ./output/sanity --unweighted -test_idx 0,1,2,3,4,5,6,7,8,9 -valid_idx 10,11,12,13,14,15,16,17,18,19

python ./src/trainer.py -dnn_depth 3 --f_dnn --u_dnn -r_valid 0.1 -b 128 -e 5 -p 2 -nn 128 -l2 0.001 -t_max 10 -dt ../SER_FEAT_EXT/h5db/ENT.MSPEC.2d.3cls.av.h5 -mt arousal:3:4:categorical_focal_loss:,valence:3:5:categorical_focal_loss: -ot ./output/sanity.txt -sm ./model/sanity -log ./output/sanity --unweighted -test_idx 0,1,2,3,4,5,6,7,8,9 -valid_idx 10,11,12,13,14,15,16,17,18,19

#sampling based (smote), multi-task learning is not supported.
python ./src/trainer.py --smote_enn -dnn_depth 3 --f_dnn --u_dnn -r_valid 0.1 -b 128 -e 5 -p 2 -nn 128 -l2 0.001 -t_max 10 -dt ../SER_FEAT_EXT/h5db/ENT.MSPEC.2d.3cls.av.h5 -mt arousal:3:4:: -ot ./output/sanity.txt -sm ./model/sanity -log ./output/sanity --unweighted -test_idx 0,1,2,3,4,5,6,7,8,9 -valid_idx 10,11,12,13,14,15,16,17,18,19
python ./src/trainer.py --smote_enn -dnn_depth 3 --f_dnn --u_dnn -r_valid 0.1 -b 128 -e 5 -p 2 -nn 128 -l2 0.001 -t_max 10 -dt ../SER_FEAT_EXT/h5db/ENT.MSPEC.2d.3cls.av.h5 -mt valence:3:5:: -ot ./output/sanity.txt -sm ./model/sanity -log ./output/sanity --unweighted -test_idx 0,1,2,3,4,5,6,7,8,9 -valid_idx 10,11,12,13,14,15,16,17,18,19

