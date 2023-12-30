# Fashion Classification using Convolutional Neural Networks

This repository contains code for a Fashion Classification project using the Fashion MNIST dataset. The goal is to classify different types of clothing items accurately.

## Dataset
The dataset used in this project is from the Fashion MNIST website. It consists of 60,000 training samples and 10,000 test samples, each containing 785 columns representing pixel values and labels.

## Problem Statement
Fashion training set consists of 70,000 images divided into 60,000 training and 10,000 testing samples. Dataset sample consists of 28x28 grayscale image, associated with a label from 10 classes. 

The 10 classes are as follows:  
- 0 => T-shirt/top
- 1 => Trouser
- 2 => Pullover
- 3 => Dress
- 4 => Coat
- 5 => Sandal
- 6 => Shirt
- 7 => Sneaker
- 8 => Bag
- 9 => Ankle boot

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255. 

## STEP 1: Importing Libraries and Data
The notebook begins by importing necessary libraries such as pandas, numpy, matplotlib, seaborn, and keras to handle data manipulation, visualization, and model building. The dataset is loaded and inspected to understand its structure and contents.

## STEP 1: Visualizing the Dataset
Exploratory visualization of the dataset is performed to gain insights into the clothing items. It includes displaying sample images, understanding pixel values, and showcasing the distribution of different clothing categories.
