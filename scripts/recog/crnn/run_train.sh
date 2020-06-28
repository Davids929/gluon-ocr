python train_crnn.py --network resnet --num-layers 34 --batch-size 128 --max-len 60 --fix-width 512 \
--short-side 32 --gpus 2 --num-workers 16 --warmup-epochs 0 --dataset-name mtwi_2018 \
--resume ./checkpoint/mtwi_2018-resnet34-crnn_best.params