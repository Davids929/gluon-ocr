# Python inference demo
This is a demo application which illustrates how to use existing Gluon-OCR models in python environments given exported JSON and PARAMS files. 

## 1. Prepare for the environment
Please refer to [quick installation](../../../doc/installation.md) to configure the Gluon-OCR operating environment.

## 2. Export models
You can refer to [detection](../../../doc/detection.md) and [recognition](../../../doc/recognition.md) to export db and crnn models.

## 3. Predict 
```bash
cd gluon-ocr/scripts/deploy/python_infer
python demo --db-model-path [db_symbol_path] --db-params-path [db_params_path] \
--crnn-model-path [crnn_symbol_path] --crnn-params-path [crnn_params_path] \
--image-path [image_path] --voc-path [dict_path] 
```
If visualize recognition results
```bash
cd gluon-ocr/scripts/deploy/python_infer
python demo --db-model-path [db_symbol_path] --db-params-path [db_params_path] \
--crnn-model-path [crnn_symbol_path] --crnn-params-path [crnn_params_path] \
--image-path [image_path] --voc-path [dict_path] --visualize --font-path [font_path]
```
