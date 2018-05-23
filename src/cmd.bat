@REM @Author: denglei
@REM @Date:   2018-05-21 18:09:03
@REM @Last Modified by:   denis
@REM Modified time: 2018-05-23 09:41:41



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

