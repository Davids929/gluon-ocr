python train_crnn.py --network resnet --num-layers 34 --batch-size 128 --max-len 60 --fix-width 384 \
--short-side 32 --gpus 2 --num-workers 16 #--resume ./ --export-model