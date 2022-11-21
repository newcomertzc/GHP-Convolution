# Appendix
The following are some supplementary experiments.

## 1. A better penalty factor for GHP Convolution
#### Update of some experimental results in the paper
| Model | Acc | Params | Macs | 
| - | :-: | :-: | :-: |
| VGG13                  | 91.64 | 129.00M | 11.28G |
| Conv-VGG13             | 92.08 | 129.00M | 11.61G |
| GHPConv-VGG13-L1-a0.01 | 92.00 | 129.00M | 11.61G |
| GHPConv-VGG13-L2-a10   | **92.19** | 129.00M | 11.61G |
| ResNet50                  | 86.67 | 23.52M | 4.04G |
| Conv-ResNet50             | 90.60 | 23.56M | 4.49G |
| GHPConv-ResNet50-L1-a0.01 | 91.01 | 23.56M | 4.49G |
| GHPConv-ResNet50-L2-a10   | **91.34** | 23.56M | 4.49G |
| ConvNext-T                  | 86.40 | 27.82M | 4.46G |
| Conv-ConvNext-T             | 87.58 | 27.84M | 4.53G |
| GHPConv-ConvNext-T-L1-a0.01 | **88.40** | 27.84M | 4.53G |
| GHPConv-ConvNext-T-L2-a10   | 87.67 | 27.84M | 4.53G |

New checkpoints are also accessible at [saved_models](saved_models).

#### Accuracy of models adopting different regularization and different penalty factors
| Model | 2021 | 2022 | 2023 | AVG |
| - | :-: | :-: | :-: | :-: |
| ResNet50                | 86.00 | 85.90 | 87.01 | 86.30 |
| Conv-ResNet50           | 90.38 | 90.03 | 90.81 | 90.41 |
| GHPConv-ResNet50-L1-a1  | 90.72 | 90.35 | 90.65 | 90.58 |
| GHPConv-ResNet50-L1-a3  | 90.39 | 90.48 | 90.62 | 90.50 |
| GHPConv-ResNet50-L1-a10 | 90.42 | 90.55 | 90.33 | 90.43 |
| GHPConv-ResNet50-L2-a1  | 90.75 | 90.53 | 90.88 | 90.72 |
| GHPConv-ResNet50-L2-a3  | 91.02 | 90.81 | 90.75 | **90.86** |
| GHPConv-ResNet50-L2-a10 | 90.74 | 90.70 | 91.14 | **90.86** |

L1-a1 indicates that the model is trained adopting L1 regularization with a penalty factor of 1, 2021 indicates that the model is trained using a random seed of 2021, and so on.
Each value in the table is an average of results run on three different GPU (Tesla-P100, RTX-3060ti and RTX-3080). 
That is, we conduct this experiments using three different random seeds on three different models of GPU (to reduce the interference of random factors).
The experimental results show that adopting L2 regularization with a penalty factor of 3 or 10 is a better choice.

#### Accuracy of different models using a new learning rate strategy
| Model | 2021 | 2022 | 2023 | AVG |
| - | :-: | :-: | :-: | :-: |
| ResNet50                | 87.84 | 87.70 | 88.25 | 87.93 |
| Conv-ResNet50           | 91.44 | 91.04 | 91.37 | 91.28 |
| GHPConv-ResNet50-L2-a3  | 92.04 | 91.65 | 91.45 | 91.71 |
| GHPConv-ResNet50-L2-a10 | 91.90 | 91.42 | 92.06 | **91.79** |

For further experiments, we adopt a new learning rate strategy different from that in the paper, i.e., we use an initial learning rate of 1e-4 and decrease it by a factor of 10 
when the number of epochs reaches 400 and 600. 
This new learning rate strategy reduces the randomness of model performance, so we only conduct this experiment on Tesla-P100 GPU. The experimental results show that
adopting a penalty factor of 10 is slightly better than using a penalty factor of 3.

## 2. Some experimental improvements
| Model | 2021 | 2022 | 2023 | AVG |
| - | :-: | :-: | :-: | :-: |
| Conv-ResNet50             | 91.44 | 91.04 | 91.37 | 91.28 |
| ConvReLU-ResNet50         | 91.59 | 91.36 | 91.37 | 91.44 |
| ConvBNReLU-ResNet50       | 91.36 | 91.35 | 91.50 | 91.40 |
| GHPConv-ResNet50-L2-a3    | 92.04 | 91.65 | 91.45 | 91.71 |
| GHPConv-ResNet50-nb-L2-a3 | 91.72 | 91.81 | 91.92 | 91.82 |
| GHPConv-ResNet50-L2-a10   | 91.90 | 91.42 | 92.06 | 91.79 |
| GHPConv-ResNet50-nb-L2-a10| 91.92 | 91.86 | 91.97 | 91.92 |

The experiment is conducted to verify some experimental improvements as follows:
+ From the perspective of a plain convolutional layer, adding an activation function (or a batchnorm layer followed and an activation function) has been widely adopted in CNNs. It may also works in this case.
+ From the perspective of a GHP convolutional layer, the bias is unnecessary. Removing it may helps. 
