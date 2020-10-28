python demo.py --model-path checkpoint/receipt-resnet50-db-symbol.json \
--params-path ./checkpoint/mtwi_2018-resnet50-db-0000.params --image-short-side 736 \
--image-path ../../data/mtwi_2018/icpr_mtwi_task2/image_test --thresh 0.3 --box-thresh 0.3 \
--gpu-id 0 --result-dir ./demo_results --visualize