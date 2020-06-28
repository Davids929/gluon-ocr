# python train_db.py --network resnet --num-layers 50 --dataset-name receipt \
# --train-img-dir /home/idcard/data/receipts/detect_data/train_img \
# --train-lab-dir /home/idcard/data/receipts/detect_data/train_lab \
# --val-img-dir /home/idcard/data/receipts/detect_data/test_img \
# --val-lab-dir /home/idcard/data/receipts/detect_data/test_lab \
# --data-shape 640 --batch-size 8 --num-workers 8 --gpus 1 --lr 0.0005 \
# --resume ./checkpoint/receipt-resnet50-db_best.params #--export-model


python train_db.py --network resnet --num-layers 50 --dataset-name mtwi_2018 \
--train-img-dir /home/idcard/data/mtwi_2018/image_train \
--train-lab-dir /home/idcard/data/mtwi_2018/txt_train \
--val-img-dir /home/idcard/data/mtwi_2018/image_1000 \
--val-lab-dir /home/idcard/data/mtwi_2018/txt_1000 \
--data-shape 640 --batch-size 8 --num-workers 0 --gpus 0 --lr 0.001 \
--resume ./checkpoint/receipt-resnet50-db_best.params #--export-model