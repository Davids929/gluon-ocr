python train_crnn.py --network resnext --num-layers 50 --batch-size 96 --max-len 50 --fix-width 384 \
--short-side 32 --gpus 1 --num-workers 8 --warmup-epochs 0 --dataset-name mtwi_2018 \
--train-data-path ../../data/mtwi_2018/image_line/train_list.txt \
--val-data-path ../../data/mtwi_2018/image_line/val_list.txt \
--voc-path ../../../gluonocr/utils/gluonocr_dict.txt \
--lr 0.001 --lr-decay 0.2 --lr-decay-epoch 60 \
#--resume ./checkpoint/mtwi_2018-resnet34-crnn_0050_0.4580.params --export-model
