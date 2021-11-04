# 2021VRDL_HW1


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python ResNet50.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python inference.py --model-file weight.pth 
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/weight.pth) 


## Results

Our model achieves the following performance on :

### [Bird Image Classification on ResNet50]

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |



