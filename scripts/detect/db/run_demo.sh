python demo.py --model-path checkpoint/receipt-resnet50-db-symbol.json \
--params-path ./checkpoint/receipt-resnet50-db-0000.params --image-short-side 736 \
--image-path ./data/20200610120910.jpg --thresh 0.3 --box-thresh 0.3 \
--visualize --gpu-id 0 --result-dir ./demo_results #--polygon