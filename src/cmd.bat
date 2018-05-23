@REM @Author: denglei
@REM @Date:   2018-05-21 18:09:03
@REM @Last Modified by:   denis
@REM Modified time: 2018-05-23 18:04:40



@REM V1
python train.py --tag base

@REM V2, add bureau info
python train.py --tag base1
python train.py --tag base1_try1    @REM, lb 0.758
python train.py --tag base1_try2
python train.py --tag base1_try3    @REM, lb 0.757
python train.py --tag base1_try4    


@REM V3, add previous app credit balance info
python train.py --tag base2
python train.py --tag base2_try1    @REM, lb 0.770
python train.py --tag base2_try2
python train.py --tag base2_try3    
python train.py --tag base2_try4    @REM, lb 0.768
python train.py --tag base2_try5    @REM, lb 0.764

@REM v4, add pos_balance info and installment-payment info
python train.py --tag base3
python train.py --tag base3_try1    @REM, lb 0.781 after adjust a few model parameters, mainly learning rate as 0.07


@REM v5, clear all dumped features and start over again
python train.py --tag good
python train.py --tag good_v1       @REM lb 0.780
python train.py --tag good_v2
python train.py --tag good_v3       @REM lb 0.782

python train.py --tag      good_v3 ^
	--max_depth            8 ^
	--num_leaves           127 ^
	--lr_rate              0.05 ^
	--early_stopping_round 100 ^
	--min_child_weight     4            @REM tag=20180523_1354, lb=0.784

python train.py --tag      good_v3_p1 ^
	--max_depth            8 ^
	--num_leaves           127 ^
	--n_estimators         5000 ^
	--lr_rate              0.03 ^
	--early_stopping_round 100 ^
	--min_child_weight     8            @REM lb=0.785, not add scale_pos_weight

python train.py --tag      good_v3_p2 ^
	--max_depth            8 ^
	--max_bin              511 ^
	--num_leaves           127 ^
	--n_estimators         5000 ^
	--lr_rate              0.01 ^
	--early_stopping_round 300 ^
	--min_child_weight     10            @REM

python train.py --tag rf_v1 --model RFClassifier     
python train.py --tag rf_v2 --model RFClassifier     @REM offline 0.7463, with max_leaf_nodes=127
python train.py --tag rf_v3 --model RFClassifier

python train.py --tag et_v2 --model ETClassifier     @REM offline bad

python train.py --tag xgb_v1 --model XGB_Classifier ^
	--learning_rate        0.01 ^
	--min_child_weight     10 ^
	--early_stopping_round 100


python train.py --model CBClassifier --tag cb_v1 
python train.py --model CBClassifier --tag cb_v2
