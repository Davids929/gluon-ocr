python demo.py --model-path ./checkpoint/mtwi_2018-resnet50-east-symbol.json \
--params-path ./checkpoint/mtwi_2018-resnet50-east-0000.params --image-short-side 640 \
--image-path ../../data/test_data/scene.jpg --score-thresh 0.5 --cover-thresh 0.1 \
--visualize --gpu-id 0 --result-dir ./test_results #--polygon

