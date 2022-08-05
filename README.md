# GHP-Convolution
Pytorch implementation for "General High-Pass Convolution: A Novel Convolutional Layer for Image Manipulation Detection".

## Train a classification Network
| Parameter                        | value                                              |
| -------------------------------- | -------------------------------------------------- |
| `--preproc`                      | `PreprocConv2d, PreprocGHPConv2d`                  |
| `--backbone`                     | `ResNet, VGG, ConvNeXt, BayarCNN`...               |
| `--backbone-func`                | `resnet50, vgg13 (function in torchvision.models)` |
| `--use-deterministic-algorithms` | `if specified, use deterministic algorithms. Otherwise, use faster algorithms` |
| `--test-only`                    | `if specified, only test the network`              |
### BayarCNN (BayarCNN_box and BayarCNN_GHP)
```
python train_classification.py --backbone BayarCNN (--use-deterministic-algorithms)
```
### ResNet (VGG and ConvNeXt)
```
python train_classification.py --backbone ResNet --backbone-func resnet50 (--use-deterministic-algorithms)
```
### Conv-ResNet (Conv-VGG and Conv-ConvNeXt)
```
python train_classification.py --backbone ResNet --backbone-func resnet50 --preproc PreprocConv2d (--use-deterministic-algorithms)
```
### GHPConv-ResNet, GHPConv-VGG and GHPConv-ConvNeXt
```
python train_classification.py --backbone ResNet --backbone-func resnet50 --preproc PreprocGHPConv2d (--use-deterministic-algorithms)
```

## Required libraries
Python 3.8.12  
Pytorch 1.8.2  
Torchvision 0.9.2  
Numpy 1.20.3  
OpenCV-Python 4.5.5.64  
matplotlib, tqdm, timm, scikit-learn, ptflops
