
python train_db.py --network resnet --num-layers 50 --dataset-name mtwi_2018 \
--train-img-dir ~/data/mtwi_2018/image_train \
--train-lab-dir ~/data/mtwi_2018/txt_train \
--val-img-dir ~/data/mtwi_2018/image_1000 \
--val-lab-dir ~/data/mtwi_2018/txt_1000 \
--data-shape 640 --batch-size 8 --num-workers 8 --gpus 1 --lr 0.001 \
--resume ./checkpoint/mtwi_2018-resnet50-db_best.params #--export-model