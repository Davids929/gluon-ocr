python demo.py --model-path checkpoint/receipt-resnet50-db-symbol.json \
--params-path ./checkpoint/receipt-resnet50-db-0000.params --image-short-side 736 \
--image-path /home/idcard/data/mtwi_2018/image_train --thresh 0.3 --box-thresh 0.3 \
--gpu-id 0 --result-dir ./demo_results --visualize