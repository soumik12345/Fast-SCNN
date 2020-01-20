# Fast-SCNN

Tensorflow 2.x implementation of [Fast-SCNN: Fast Semantic Segmentation Network](https://arxiv.org/abs/1902.04502).

## Instructions for setting up Cityscapes

- ```git clone -b cityscapes https://github.com/soumik12345/Fast-SCNN```
- ```cd ./Fast-SCNN/data```
- ```bash cityscapes.sh```

## Project Directory Structure

<code><pre>.
├── README.md
├── data
│   ├── cityscapes.sh
│   ├── train_imgs
│   ├── train_labels
│   ├── val_imgs
│   └── val_labels
├── model.png
├── requirements.txt
├── src
│   ├── blocks.py
│   ├── cityscapes.py
│   ├── model.py
│   └── utils.py
└── train.py

</code></pre>

## Model Architecture

![](./model.png)