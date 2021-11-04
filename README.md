# 2021VRDL_HW1

## Environment

```
Ubuntu 16.04.5 LTS (GNU/Linux 4.15.0-39-generic x86_64)
```

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Furthermore, please put the datas you need in the path of the py files.

## Training

To train the model(s) in the paper, run this command:

```train
python ResNet50.py 
```

## Testing

To evaluate my model on ImageNet, run:

```eval
python inference.py --model-file weight.pth 
```

## Weight for Training Model

You can download the file here:

- [The file of weight](https://drive.google.com/file/d/1EhhSuLb4FHcRADGh7Fi_ute492mAQavk/view?usp=sharing) 


## Results

Our model achieves the following performance on :

### [Bird Image Classification on ResNet50]

| Model name         | ResNet50  | 
| ------------------ |---------- | 
| Score              |   60.4%   | 



