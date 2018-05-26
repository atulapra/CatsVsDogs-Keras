# Cats Vs Dogs in Keras

* Kaggle's Cats Vs Dogs Redux challenge is a great way for deep learning learners to quickly implement CNNs, transfer learning, fine-tuning,etc and experiment with their knowledge.
* This repository is a collection of the gists referenced in [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) keras blog post, along with some corrections to remove common errors and to ensure compatibility in both Python 2 and 3. 

## Setup

* Clone the repository using `git clone https://github.com/atulapra/CatsVsDogs-Keras.git`.
* Go into the cloned folder with `cd CatsVsDogs-Keras`.
* Download the dataset from [here](https://anonfile.com/y0h9D3edbd/data.tar.gz) and unzip it in the current folder.
* Download the vgg16 weights from [here.](https://anonfile.com/Gbh8D1efba/vgg16_weights.h5)
* I have trained these models on Nvidia GeForce GTX 1050 Ti GPU, and hence my training time would be quite low as compared to CPU. So, if you are going to train this in CPU, it might take a susbstantial amount of time.

## Usage

* The program `conv.py` employs a 3 layer convolutional neural network for classification. Just run `python conv.py`. The weights for this model will be saved as `first_try.h5`. Training takes ~15s per epoch. We get a test accuracy of ~80%.
* The program `bottleneck.py` employs a VGG16 network with FC layers retrained. Run `python bottleneck.py`. The weights will be saved as `bottleneck_fc_model.h5`. Training takes only 1s per epoch. We get a test accuracy of ~90%.
* The program `finetune.py` is used to fine tune the model trained above. Run `python finetune.py`. Training takes ~85s per epoch.

* Note that `finetune.py` can be run only after you have saved the weights `bottleneck_fc_model.h5`, which is obtained by running `bottleneck.py`.



