## TEXT RECOGNITION

This section uses the icdar2015 dataset as an example to introduce the training, evaluation, and testing of the recognition model in Gluon-OCR.

### DATA PREPARATION

Downloading icdar2015 dataset from [official website](https://rrc.cvc.uab.es/?ch=4&com=downloads).

Decompress the downloaded dataset to the working directory, assuming it is decompressed under gluon-ocr/train_data/.
After decompressing the data set and downloading the annotation file, gluon-ocr/train_data/ has two folders and two files, which are:
```
/gluon-ocr/train_data/icdar2015/
  └─ ch4_train_word_imgs/                Training image of icdar dataset
  └─ ch4_test_word_imgs/                 Testing image of icdar dataset
  └─ train_icdar2015_label.txt           Training annotation of icdar dataset
  └─ test_icdar2015_label.txt/           Testing annotation of icdar dataset
```
The provided annotation file format is as follow, seperated by "\t", image file path is absolute path:
```
" Image file path                         Image annotation"
ch4_test_word_imgs/img_1.png                   JOINT
ch4_test_word_imgs/img_2.png                   yourself
        ...                                     ...
```

A dictionary ({word_dict_name}.txt) needs to be provided so that when the model is trained, all the characters that appear can be mapped to the dictionary index.

Therefore, the dictionary needs to contain all the characters that you want to be recognized correctly. {word_dict_name}.txt needs to be written in the following format and saved in the `utf-8` encoding format:

```
l
d
a
...
```

`gluonocr/utils/gluonocr_dict.txt` is a Chinese dictionary with 7434 characters.

`gluonocr/utils/ic15_dict.txt` is an English dictionary with 36 characters.

### TRAINING
Training CRNN.
```shell
cd gluon-ocr/scripts/recog/crnn
python train_crnn.py --network resnet --num-layers 34 --batch-size 300 --dataset-name icdar15 \
--max-len 30 --fix-width 384 --short-side 32 --gpus 2,3 --num-workers 16 --warmup-epochs 0 \
--lr 0.001 --lr-decay 0.1 --lr-decay-epoch 60 --wd 0.0005 --voc-path gluonocr/utils/ic15_dict.txt \
--train-data-path gluon-ocr/train_data/icdar2015/train_icdar2015_label.txt \
--val-data-path gluon-ocr/train_data/icdar2015/test_icdar2015_label.txt \
--save-prefix ./checkpoint
```

Export model
```shell
cd gluon-ocr/scripts/recog/crnn
python train_crnn.py --network resnet --num-layers 34 --batch-size 300 --dataset-name icdar15 \
--max-len 30 --fix-width 384 --short-side 32 --gpus 2,3 --num-workers 16 --warmup-epochs 0 \
--lr 0.001 --lr-decay 0.1 --lr-decay-epoch 60 --wd 0.0005 --voc-path gluonocr/utils/ic15_dict.txt \
--train-data-path gluon-ocr/train_data/icdar2015/train_icdar2015_label.txt \
--val-data-path gluon-ocr/train_data/icdar2015/test_icdar2015_label.txt \
--save-prefix ./checkpoint --resume ./checkpoint/icdar15-resnet34-crnn_best.params --export-model
```

gluon-ocr/scripts/recog/crnn/checkpoint will be generated two files, which are:

```
  └─ icdar15-resnet34-crnn-0000.params          DBNet params file
  └─ icdar15-resnet34-crnn-symbol.json          DBNet model file
```

## TEST
Test the recognition result on a single image:
```shell
cd gluon-ocr/scripts/recog/crnn
python demo.py --model-path ./checkpoint/icdar15-resnet34-crnn-symbol.json \
--params-path ./checkpoint/icdar15-resnet34-crnn-0000.params \
--voc-path gluonocr/utils/ic15_dict.txt \
--image-path ./doc/imgs/word_1.png \
--gpu-id 1 
```