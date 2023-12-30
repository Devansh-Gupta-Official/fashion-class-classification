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

## Model Workflow

### STEP 1: Importing Libraries and Data
The notebook starts by importing necessary libraries:

- pandas: For data manipulation
- numpy: For numerical operations
- matplotlib and seaborn: For data visualization
- keras: For building and training the neural network model
The dataset is loaded into pandas DataFrames using pd.read_csv() from respective CSV files: fashion-mnist_train.csv and fashion-mnist_test.csv. The data is then inspected to understand its structure using methods like head(), shape, and info().

### STEP 2: Visualizing the Dataset
Exploratory visualization is performed to gain insights into the dataset:

- Displaying sample images using matplotlib.pyplot.imshow()
- Understanding pixel values and their distribution
- Showcasing the distribution of different clothing categories using plots such as histograms or count plots from seaborn

### STEP 3: Model Training
A Convolutional Neural Network (CNN) architecture is implemented using Keras. The model consists of convolutional layers, pooling layers, and fully connected layers. The model is compiled with appropriate loss, optimizer, and metrics for training. The model is trained using the Adam optimizer and sparse categorical cross-entropy loss for 50 epochs. During training, accuracy steadily increases, and validation accuracy stabilizes around 88%.

**Preprocessing Data**
Data preprocessing steps include:

- Normalization of pixel values to bring them into a range suitable for model training
- Splitting data into training, validation, and testing sets using train_test_split from sklearn
- Reshaping images into a format suitable for CNN input (in this case, (28, 28, 1) for grayscale images)

  
**Model Architecture**
The CNN model architecture involves:

- Input layer: 28x28x1 (grayscale)
- Convolutional layers with ReLU activation
- Max Pooling layers for down-sampling
- Dense layers with softmax activation for classification

  
**Model Training**
The model is compiled using the Adam optimizer and sparse_categorical_crossentropy loss function. Training is performed using model.fit() method with the training data, validation data, batch size, and number of epochs.

### STEP 4: Evaluating the Model
After training, the model's performance is evaluated using the test dataset:

- Accuracy metrics on the test set using model.evaluate()
- Prediction and comparison of predicted classes with true classes
- Detailed evaluation metrics including precision, recall, and F1-score using classification_report from sklearn.metrics

## Results
The model achieved an accuracy of **88.12%** on the test set. The classification report provides detailed metrics including precision, recall, and F1-score for each clothing class.

**Visualizations**
The README includes visualizations of:

- Random images from the dataset to showcase clothing items
- Confusion matrix providing insights into classification performance
- Classification report for detailed evaluation metrics


## Usage
To run this project:

1. Clone the Repository
Clone this repository to your local machine using the following command:
```
git clone https://github.com/Devansh-Gupta-Official/fashion-class-classification.git
```

2. Set Up Environment
Make sure you have Python installed. Create a virtual environment to manage dependencies:
```
# Navigate to the project directory
cd fashion-class-classification
```
3. Install Dependencies
Install the necessary libraries specified in the requirements.txt file:
```
pip install -r requirements.txt
```
4. Run the Jupyter Notebook
Launch Jupyter Notebook or JupyterLab and open the fashion_class.ipynb notebook:
```
jupyter notebook
```

5. Execute Notebook Cells
Execute each cell in the notebook sequentially. The notebook contains detailed explanations, code, and comments explaining the entire workflow:

- Importing and exploring the dataset
- Preprocessing steps such as normalization and data splitting
- Building and training the Convolutional Neural Network (CNN) model
- Evaluating the model's performance on the test set
- Visualizing results, including images, confusion matrix, and classification repor

The notebook contains detailed code implementation, explanations, and comments, enabling users to understand the process and replicate the project.
