#!/bin/bash

python InstanceCLS_train.py --max-epoch 120 --eval-step 1 --gpu-devices '3,2,1,0' --d air --width 224 --height 224 --a squeezenetv2 --save-dir log/squeezenetv2_sch_224
python InstanceCLS_train.py --max-epoch 120 --eval-step 1 --gpu-devices '3,2,1,0' --d air --width 224 --height 224 --a shufflenetv2 --save-dir log/shufflenetv2_sch_224
python InstanceCLS_train.py --max-epoch 120 --eval-step 1 --gpu-devices '3,2,1,0' --d air --width 224 --height 224 --a mobilenetv2 --save-dir log/mobilenetv2_sch_224
python InstanceCLS_train.py --max-epoch 120 --eval-step 1 --gpu-devices '3,2,1,0' --d air --width 224 --height 224 --a densenet121 --save-dir log/densenet121_sch_224
python InstanceCLS_train.py --max-epoch 120 --eval-step 1 --gpu-devices '3,2,1,0' --d air --width 224 --height 224 --a resnet50 --save-dir log/resnet50_sch_224
python InstanceCLS_train.py --max-epoch 120 --eval-step 1 --gpu-devices '3,2,1,0' --d air --width 224 --height 224 --a vggnet16 --save-dir log/vggnet16_sch_224



