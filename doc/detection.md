# TEXT DETECTION

This section uses the icdar2015 dataset as an example to introduce the training, evaluation, and testing of the detection model in Gluon-OCR.

## Data preparation

Downloading icdar2015 dataset from [official website](https://rrc.cvc.uab.es/?ch=4&com=downloads).

Decompress the downloaded dataset to the working directory, assuming it is decompressed under gluon-ocr/train_data/. 
 After decompressing the data set and downloading the annotation file, gluon-ocr/train_data/ has four folders, which are:
```
/gluon-ocr/train_data/icdar2015/
  └─ icdar_c4_train_imgs/         Training image of icdar dataset
  └─ icdar_c4_train_labs/         Training label of icdar dataset
  └─ ch4_test_images/             Testing image of icdar dataset
  └─ ch4_test_labels/             Testing label of icdar dataset
```

## Training

Training DBNet.
```shell
cd gluon-ocr/scripts/detect/db
python train_db.py --network resnet --num-layers 50 --dataset-name icdar15 \
--train-img-dir ../../data/icdar2015/icdar_c4_train_imgs \
--train-lab-dir ../../data/icdar2015/icdar_c4_train_labs \
--val-img-dir ../../data/icdar2015/ch4_test_images \
--val-lab-dir ../../data/icdar2015/ch4_test_labels \
--data-shape 640 --batch-size 4 --num-workers 8 --gpus 1 --lr 0.001 \
--save-prefix ./checkpoint
```

## Export model
```shell
cd gluon-ocr/scripts/detect/db
python train_db.py --network resnet --num-layers 50 --dataset-name icdar15 \
--data-shape 640 --gpus 1 --save-prefix ./checkpoint --export-model \
--resume ./checkpoint/icdar15-resnet50-db_best.params
```
gluon-ocr/scripts/detect/db/checkpoint will be generated two files, which are:
```
  └─ icdar15-resnet50-db-0000.params          DBNet params file
  └─ icdar15-resnet50-db-symbol.json          DBNet model file
```

<!-- EVALUATION -->

## Test
Test the detection result on a single image:
```shell
cd gluon-ocr/scripts/detect/db
python demo.py --model-path ./checkpoint/icdar15-resnet50-db-symbol.json \
--params-path ./checkpoint/icdar15-resnet50-db-0000.params --image-short-side 736 \
--image-path ../../../doc/imgs/img_10.jpg --thresh 0.3 --box-thresh 0.3 \
--gpu-id 0 --result-dir ./demo_results --visualize
```