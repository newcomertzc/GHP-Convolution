# GHP-Convolution
Pytorch implementation for "General High-Pass Convolution: A Novel Convolutional Layer for Image Manipulation Detection".

## 1. How to train a classification network
| Parameter                        | value                                                                              |
| -------------------------------- | ---------------------------------------------------------------------------------- |
| `--data-path`                    | `the path to training set`                                                         |
| `--data-val-path`                | `the path to validation set`                                                       |
| `--preproc`                      | `PreprocConv2d, PreprocGHPConv2d, PreprocIdentity(just a placeholder module)`      |
| `--backbone`                     | `ResNet, VGG, ConvNeXt, BayarCNN, BayarCNN_box, BayarCNN_GHP`...                   |
| `--backbone-func`                | `resnet50, vgg13, convnext_tiny (function of torchvision.models or core.convnext)` |
| `--use-deterministic-algorithms` | `if specified, use deterministic algorithms. Otherwise, use faster algorithms`     |
| `--test-only`                    | `if specified, only test the network`                                              |
### BayarCNN (BayarCNN_box, BayarCNN_GHP)
```
python train_classification.py --backbone BayarCNN
```
### ResNet (VGG, ConvNeXt)
```
python train_classification.py --backbone ResNet --backbone-func resnet50
```
(`backbone-func` can be `resnet34, vgg16...` (function provided by torchvision) and `convnext_tiny, convnext_small...` (function in [this file](core/convnext.py)))
### Conv-ResNet (Conv-VGG, Conv-ConvNeXt)
```
python train_classification.py --backbone ResNet --backbone-func resnet50 --preproc PreprocConv2d
```
### GHPConv-ResNet (GHPConv-VGG and GHPConv-ConvNeXt)
```
python train_classification.py --backbone ResNet --backbone-func resnet50 --preproc PreprocGHPConv2d
```

## 2. How to train a segmentation network
| Parameter                        | value                                                                              |
| -------------------------------- | ---------------------------------------------------------------------------------- |
| `--coco-path`                    | `the path to COCO training set`                                                    |
| `--coco-ann-path`                | `the path to the annotation file 'instances_train2017.json'`                       |
| `--data-val-path`                | `the path to validation set`                                                       |
| `--backbone`                     | `DeepLabV3_ResNet, FCN_ResNet, FCN_VGG_8s, FCN_VGG_16s, FCN_VGG_8s`                |
| `--pretrained`                   | `the path to your pretrained classification network checkpoint`                    |
| `--replace-stride-with-dilation` | `adjust the dilation of ResNet`     |
| `--use-deterministic-algorithms` | `if specified, use deterministic algorithms. Otherwise, use faster algorithms`     |
| `--test-only`                    | `if specified, only test the network`                                              |

A pretrained classification network is required as the backbone network to train the segmentation network.
### FCN-VGG-8s (FCN-VGG-16s, FCN-VGG-32s)
```
python train_segmentation.py --pretrained your_pretrained_checkpoint --backbone FCN_VGG_8s
```
### FCN-ResNet-8s (FCN-ResNet-16s, DeepLabV3-ResNet-8s)
```
python train_segmentation.py --pretrained your_pretrained_checkpoint --backbone FCN_ResNet --replace-stride-with-dilation 0 1 1
```
(For `replace-stride-with-dilation`, `0 1 1` indicates XXX-ResNet-8s, `0 0 1` indicates XXX-ResNet-16s and `0 0 0` indicates XXX-ResNet-32s.)

## 3. Required libraries
`Python == 3.8.12`  
`Pytorch == 1.8.2`  
`Torchvision == 0.9.2`  
`Pillow == 9.0.1`  
`Numpy == 1.20.3`  
`OpenCV-Python == 4.5.5.64`  
`Matplotlib, Scikit-learn, tqdm, timm, ptflops`
