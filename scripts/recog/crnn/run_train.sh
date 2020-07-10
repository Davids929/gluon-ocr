# python train_crnn.py --network resnet --num-layers 34 --batch-size 128 --max-len 50 --fix-width 320 \
# --short-side 32 --gpus 1 --num-workers 8 --warmup-epochs 0 --dataset-name receipt \
# --resume ./checkpoint/receipt-resnet34-crnn_best.params #--export-model

python train_crnn.py --network resnet --num-layers 34 --batch-size 100 --max-len 50 --fix-width 448 \
--short-side 32 --gpus 1 --num-workers 8 --warmup-epochs 0 --dataset-name scene \
--lr 0.0001 --lr-decay 0.2 --lr-decay-epoch 60 \
--resume ./checkpoint/scene-resnet34-crnn_0070_0.4506.params #--export-model
