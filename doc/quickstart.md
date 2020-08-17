# Quick start of Chinese OCR model

## 1. Prepare for the environment

Please refer to [quick installation](./installation.md) to configure the GluonOCR operating environment.

## 2. Export models
  example: export crnn model
  ```
    cd gluon-ocr/scripts/recog/crnn
    python train_crnn.py --network resnet --num-layers 34 --dataset-name icdar19 --resume ./checkpoint/icdar19-resnet34-crnn_best.params
  ```   

## 3. Inference and display
  - text detection
      - Predict with pre-trained DB models
      ```
        cd gluon-ocr/scripts/detect/db
        sh run_demo.sh
      ```
  - text recognition
      - Predict with pre-trained CRNN models
      ```
        cd gluon-ocr/scripts/recog/crnn
        sh run_demo.sh
      ```