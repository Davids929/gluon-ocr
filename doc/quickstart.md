# Quick start of Chinese OCR model

## 1. Prepare for the environment

Please refer to [quick installation](./installation.md) to configure the Gluon-OCR operating environment.

## 2. Download models
|Name| Introduction | Detection model |Recognition model|
|-|-|-|-|
|chinese_db_crnn|Universal Chinese OCR model|[inference model]() / [pretrained model]()|[inference model]() / [pretrained model]()|
|chinese_db_rare|Universal Chinese OCR model|[inference model]() / [pretrained model]()|[inference model]() / [pretrained model]()|


## 3. Export models
  Export crnn model
  ```
    cd gluon-ocr/scripts/recog/crnn
    python train_crnn.py --network resnet --num-layers 34 --dataset-name icdar19 --resume ./checkpoint/icdar15-resnet34-crnn_best.params
  ```   

## 4. Inference and display
  - Text detection
      - Predict with pre-trained DB models
      ```
        cd gluon-ocr/scripts/detect/db
        sh run_demo.sh
      ```
  - Text recognition
      - Predict with pre-trained CRNN models
      ```
        cd gluon-ocr/scripts/recog/crnn
        sh run_demo.sh
      ```