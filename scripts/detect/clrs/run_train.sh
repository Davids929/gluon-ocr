python train_clrs.py --network resnet --num-layers 50 --dataset-name mtwi_2018 \
--train-img-dir /home/sw/data/mtwi_2018/image_train \
--train-lab-dir /home/sw/data/mtwi_2018/txt_train \
--val-img-dir /home/sw/data/mtwi_2018/image_1000 \
--val-lab-dir /home/sw/data/mtwi_2018/txt_1000 \
--data-shape 768 --batch-size 4 --num-workers 8 --gpus 3 \
--lr 0.001 --lr-decay 0.2 --warmup-epochs 2 \
--log-interval 10 \
#--resume ./checkpoint/mtwi_2018-resnet50-clrs_best.params #--export-model