python demo.py --model-path ./checkpoint/mtwi_2018-resnet50-clrs-symbol.json \
--params-path ./checkpoint/mtwi_2018-resnet50-clrs-0000.params --image-size 512 \
--image-path ~/gluon-ocr/doc/imgs/11.jpg --seg-thresh 0.6 --box-thresh 0.3 \
--gpu-id 2 --result-dir ./demo_results --visualize