# handgesture.github.io
Repository for a browser based tensorflow model to classify hand gestures.

## Introduction

We make use of trasfer learning and of [MobileNet](https://arxiv.org/abs/1704.04861), a set of light and efficient CNNs designed specifically for mobile and embedded computer vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks.

## How to use

Simply, go to the [webpage](handgesture.github.io). 
It might be required to allow the browser to access your webcam.
Hence, you build a little dataset (50/60 images per class should be enough).

### Example of dataset construction

You put your hand in front of the camera, indicating for instance 2.
Click on the `two` button would add the image and the corresponding label to the dataset.

### Training
Once you are happy with the examples you have added, train your network.
A browser alert would notify when training is done, allowing you starting the predictions.

### Predictions
Again, you put your hand in front of the camera, miming the gesture of three, and you should visualise the message _I see three_.