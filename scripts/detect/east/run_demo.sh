python demo.py --model-path ./checkpoint/receipt-resnet50-east-symbol.json \
--params-path ./checkpoint/receipt-resnet50-east-0000.params --image-short-side 736 \
--image-path ./test_data/TBAL1901040401201602005.jpg --score-thresh 0.5 --cover-thresh 0.1 \
--visualize --gpu-id 0 --result-dir ./test_results #--polygon

