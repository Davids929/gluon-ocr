python train_att_model.py --network resnet --num-layers 34 --batch-size 48 --max-len 60 \
--bucket-mode --short-side 32 --gpus 1 --num-workers 8 --start-epoch 58 \
--resume ./checkpoint/att-model_resnet34_best.params --warmup-epochs 0 #--export-model