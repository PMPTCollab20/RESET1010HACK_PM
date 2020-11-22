# RESET1010HACK_PM

## [Android app] for Generic Object detection using PyTorch Mobile model

## [Insect Classifier] Python PyTorch Trainer for BEE vs ANT

### Run Insect CLASSIFIER

python classifier/main.py

![Result](results-classifier/50epoch/latest/Figure_2.png)

located in folder [results-classifier][run-no]

Augmented training on the pre-trained model [RESNET18](https://pytorch.org/hub/pytorch_vision_resnet/)

[Based on beta version tutorial of Pytorch](https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html)

### Setup Pytorch Framework with NVIDIA GPU

Recommend using Anaconda Environment with Python 3.7

pip install numpy
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html
