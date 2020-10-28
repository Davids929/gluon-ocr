python train_att_model.py --network resnet --num-layers 34 --batch-size 48 --max-len 60 \
--bucket-mode --short-side 32 --gpus 3 --num-workers 8 --start-epoch 0 --dataset-name mtwi_2018 \
--train-data-path ../../data/mtwi_2018/image_line/train_list.txt \
--val-data-path ../../data/mtwi_2018/image_line/val_list.txt \
--voc-path ../../../gluonocr/utils/gluonocr_dict.txt
#--resume ./checkpoint/att-model_resnet34_best.params --warmup-epochs 2 #--export-model