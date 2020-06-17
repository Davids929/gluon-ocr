python train_east.py --network resnet --num-layers 50 --dataset-name receipt \
--train-img-dir /home/idcard/data/receipts/detect_data/train_img \
--train-lab-dir /home/idcard/data/receipts/detect_data/train_lab \
--val-img-dir /home/idcard/data/receipts/detect_data/test_img \
--val-lab-dir /home/idcard/data/receipts/detect_data/test_lab \
--data-shape 640 --batch-size 8 --num-workers 8 --gpus 3 \
--lr 0.001 \
--resume ./checkpoint/receipt-resnet50-east_best.params --warmup-epochs 0 --export-model