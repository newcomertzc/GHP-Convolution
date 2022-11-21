# GHP-Convolution
Pytorch implementation for "General High-Pass Convolution: A Novel Convolutional Layer for Image Manipulation Detection".

## 1. To train a image manipulaion classification network
| Parameter                        | value                                                                              |
| -------------------------------- | ---------------------------------------------------------------------------------- |
| `--data-path`                    | `training set path`                                                                |
| `--data-val-path`                | `validation set path`                                                              |
| `--preproc`                      | `PreprocConv2d or PreprocGHPConv2d. If not specified, use a placeholder (PreprocIdentity) instead.`  |
| `--backbone`                     | `ResNet, VGG, ConvNeXt, BayarCNN, BayarCNN_box, BayarCNN_GHP...`                   |
| `--backbone-func`                | `resnet50, vgg13, convnext_tiny (function of torchvision.models and core.convnext used to create models)` |
| `--use-deterministic-algorithms` | `if specified, use deterministic algorithms (but slower).`                         |
| `--test-only`                    | `if specified, only test the network on validation set.`                           |
| `--reproduce`                    | `if specified, use some deprecated functions and training settings to reproduce the experimental results.` |
### BayarCNN (or BayarCNN_box, BayarCNN_GHP)
```
python train_classification.py --backbone BayarCNN
```
### ResNet (or VGG, ConvNeXt)
```
python train_classification.py --backbone ResNet --backbone-func resnet50
```
(`backbone-func` can be `resnet34, vgg16...` (function provided by torchvision) and `convnext_tiny, convnext_small...` (function of [core.convnext](core/convnext.py)))
### Conv-ResNet (or Conv-VGG, Conv-ConvNeXt)
```
python train_classification.py --backbone ResNet --backbone-func resnet50 --preproc PreprocConv2d
```
### GHPConv-ResNet (or GHPConv-VGG and GHPConv-ConvNeXt)
```
python train_classification.py --backbone ResNet --backbone-func resnet50 --preproc PreprocGHPConv2d
```

## 2. To train a image manipulation segmentation network
| Parameter                        | value                                                                              |
| -------------------------------- | ---------------------------------------------------------------------------------- |
| `--coco-path`                    | `COCO training set path`                                                           |
| `--coco-ann-path`                | `path to the annotation file 'instances_train2017.json'`                           |
| `--data-val-path`                | `validation set path`                                                              |
| `--backbone`                     | `DeepLabV3_ResNet, FCN_ResNet, FCN_VGG_8s, FCN_VGG_16s, FCN_VGG_8s`                |
| `--pretrained`                   | `path to the pretrained classification network checkpoint`                         |
| `--replace-stride-with-dilation` | `a parameter used to adjust the dilation of ResNet`                                |
| `--use-deterministic-algorithms` | `if specified, use deterministic algorithms. Otherwise, use faster algorithms`     |
| `--test-only`                    | `if specified, only test the network`                                              |

A pretrained classification network is required to act as the backbone network of the segmentation network.
### FCN-VGG-8s (or FCN-VGG-16s, FCN-VGG-32s)
```
python train_segmentation.py --pretrained your_pretrained_checkpoint --backbone FCN_VGG_8s
```
### FCN-ResNet-8s (or FCN-ResNet-16s, DeepLabV3-ResNet-8s)
```
python train_segmentation.py --pretrained your_pretrained_checkpoint --backbone FCN_ResNet --replace-stride-with-dilation 0 1 1
```
(For the parameter `replace-stride-with-dilation`, `0 1 1` means stride = 8 (FCN-ResNet-8s), `0 0 1` means stride = 16 and `0 0 0` means stride = 32)

## 3. Appendix
 See [appendix.md](appendix.md).
 
## 4. Required libraries 
`python == 3.8.10`  
`pytorch == 1.9.1`  
`torchvision == 0.10.1`  
`pillow == 9.0.1`  
`numpy == 1.20.3`  
`opencv-python == 4.5.3.56`  
`scikit-learn == 0.24.2`  
`pycocotools == 2.0.4`  
`ptflops == 0.6.8`  
`tqdm == 4.64.0`  
