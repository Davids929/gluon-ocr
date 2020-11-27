python train_att_model.py --network resnet --num-layers 34 --batch-size 48 --max-len 60 \
--bucket-mode --short-side 32 --gpus 3 --num-workers 8 --start-epoch 0 --dataset-name mtwi_2018 \
--train-data-path ~/data/mtwi_2018/image_line/train_list.txt \
--val-data-path ~/data/mtwi_2018/image_line/val_list.txt \
--voc-path ../../../gluonocr/utils/gluonocr_dict.txt \
--lr 0.001 --lr-decay 0.1 --lr-decay-epoch 60,80 --epochs 100 --wd 0.0005 \
#--resume ./checkpoint/mtwi_2018-resnet34-att-model_best.params --export-model