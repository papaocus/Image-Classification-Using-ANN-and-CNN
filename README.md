# Image-Classification-Using-ANN-and-CNN
This code demonstrates the implementation of a deep learning model for image classification using the CIFAR-10 dataset. The code uses the TensorFlow library and consists of two parts: a basic artificial neural network (ANN) and a convolutional neural network (CNN).

The code begins by importing the necessary libraries, including TensorFlow, datasets, layers, models, matplotlib, and numpy. It then loads the CIFAR-10 dataset, which consists of 50,000 training images and 10,000 test images of size 32x32 with three color channels (RGB).

The dataset is divided into training and test sets, and the shape of the training and test data is displayed. The labels are also reshaped for further processing. A list of class names is defined for visualization purposes.

The code includes a function, plot_sample, which plots a sample image from the dataset along with its corresponding label. Some sample images are plotted using this function.

Next, the pixel values of the images are normalized by dividing them by 255.0 to bring them within the range of 0 to 1.

The first model implemented is an ANN with three dense layers. The model is compiled with the stochastic gradient descent (SGD) optimizer and sparse categorical cross-entropy loss. It is then trained on the normalized training data with the corresponding labels for 20 epochs.

After training, the model is evaluated using the test data, and a classification report is printed, showing metrics such as precision, recall, and F1-score for each class.

The second model implemented is a CNN. It consists of two convolutional layers with ReLU activation, followed by max-pooling layers. The flattened output is then passed through two dense layers with ReLU and softmax activation functions. The model is compiled with the Adam optimizer and trained on the training data for 10 epochs.

After training, the CNN model is evaluated on the test data, and its performance is displayed by printing the loss and accuracy.

Predictions are made on the test data using the CNN model, and the predicted classes are extracted. Some sample predictions are displayed along with their corresponding true labels.

Finally, the plot_sample function is used to visualize some test images along with their true labels and predicted classes.

This code provides a basic implementation of an ANN and a CNN for image classification using the CIFAR-10 dataset. It serves as a starting point for further exploration and improvement in deep learning-based image classification tasks.
