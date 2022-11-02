# Appendix
The following are some supplementary experiments.

## 1. The optimal regularization approach and penalty factor for GHP Convolution
#### Accuracy of different models
| Model | 2021 | 2022 | 2023 | AVG |
| - | :-: | :-: | :-: | :-: |
| ResNet50                | 86.00 | 85.90 | 87.01 | 86.30 |
| Conv-ResNet50           | 90.38 | 90.03 | 90.81 | 90.41 |
| GHPConv-ResNet50-L1-a1  | 90.72 | 90.35 | 90.65 | 90.58 |
| GHPConv-ResNet50-L1-a3  | 90.39 | 90.48 | 90.62 | 90.50 |
| GHPConv-ResNet50-L1-a10 | 90.42 | 90.55 | 90.33 | 90.43 |
| GHPConv-ResNet50-L2-a1  | 90.75 | 90.53 | 90.88 | 90.72 |
| GHPConv-ResNet50-L2-a3  | 91.02 | 90.81 | 90.75 | 90.86 |
| GHPConv-ResNet50-L2-a10 | 90.74 | 90.70 | 91.14 | 90.86 |

L1-a1 indicates that the model is trained adopting L1 regularization with a penalty factor of 1, 2021 indicates the model is trained using a random seed of 2021, and so on.
Each value in the table above is an average of results run on three different models of GPU (Tesla-P100, RTX-3060ti and RTX-3080). 
That is, we conduct this experiments using three different random seeds on three different models of GPU to reduce the interference of random factors.
This range of 1 - 10 is determined based on some preliminary experimental results. 
The experimental results show that adopting L2 regularization with a penalty factor of 3 or 10 is the best choice.

#### Accuracy of different models using a new learning rate strategy
| Model | 2021 | 2022 | 2023 | AVG |
| - | :-: | :-: | :-: | :-: |
| ResNet50                | 87.84 | 87.70 | 88.25 | 87.93 |
| Conv-ResNet50           | 91.44 | 91.04 | 91.37 | 91.28 |
| GHPConv-ResNet50-L2-a3  | 92.04 | 91.65 | 91.45 | 91.71 |
| GHPConv-ResNet50-L2-a10 | 91.90 | 91.42 | 92.06 | 91.79 |

For further experiments, we adopt a new learning rate strategy different from the paper, i.e., we use an initial learning rate of 1e-4 and decaying it by a factor of 10 
when the number of epoch reaches 400 and 600. 
This new learning rate strategy reduces the randomness of model performance, so we only conduct this experiment on Tesla-P100 GPU. The experimental results show that
using a penalty factor of 10 is slightly better than using a penalty factor of 3, and the former is more stable.

## 2. Adding an activation layer or remove the bias of convolution layer
| Model | 2021 | 2022 | 2023 | AVG |
| - | :-: | :-: | :-: | :-: |
| Conv-ResNet50           | 91.44 | 91.04 | 91.37 | 91.28 |
| ConvReLU-ResNet50       | 91.59 | 91.36 | 91.37 | 91.44 |
| ConvBNReLU-ResNet50     | 91.36 | 91.35 | 91.50 | 91.40 |
| GHPConv-ResNet50-L2-a3  | 92.04 | 91.65 | 91.45 | 91.71 |
| GHPConv-ResNet50-L2-a3  | 91.72 | 91.81 | 91.92 | 91.82 |
| GHPConv-ResNet50-L2-a10 | 91.90 | 91.42 | 92.06 | 91.79 |
| GHPConv-ResNet50-L2-a10 | 91.92 | 91.86 | 91.97 | 91.92 |

This experiment is conducted to verify the following two ideas:
+ For a plain convolutional layer, adding an activation function (or a batchnorm layer and an activation function) after it has been shown to improve the performance of the model in most cases
  verified to improve its performance. It's also very likely to work in this case.
+ For a GHP convolutional layer, the bias is unnecessary or even detrimental, as it's designed to simulate a series of high-pass filters. Removing the bias may helps. 
