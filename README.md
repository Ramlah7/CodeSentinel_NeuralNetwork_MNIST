# NeuralNetwork_MNIST

### Task 3: Build and Train a Neural Network with Keras/TensorFlow

This repository contains the solution for **Task 3** of the Code Sentinel "Artificial Intelligence" Virtual Internship. The project's goal was to build and train a deep learning model for image classification using the MNIST dataset.

### Project Overview

The project successfully implements a fully connected neural network to classify handwritten digits (0-9). The code, written in a Jupyter Notebook, covers all the essential steps of a machine learning workflow: data loading, preprocessing, model building, training, and evaluation. The model achieved a high level of accuracy and demonstrates a strong understanding of deep learning fundamentals.

### Key Concepts and Implementation

* **Data Preparation:** The MNIST dataset was loaded and preprocessed by normalizing the pixel values to a `[0, 1]` range. The 2D images ($28 \times 28$) were then flattened into 1D vectors for the dense network.
* **One-Hot Encoding:** The integer labels were converted to a one-hot encoded format, which is required for the chosen loss function.
* **Model Architecture:** A `Sequential` Keras model was built with:
    * An input layer with a shape of 784.
    * Two `Dense` hidden layers, each with 128 neurons and a `relu` activation function.
    * A `Dropout` layer with a rate of 0.25 to prevent overfitting.
    * A final `Dense` output layer with 10 neurons and a `softmax` activation to output class probabilities.
* **Training and Evaluation:** The model was compiled with the `adam` optimizer and `categorical_crossentropy` loss. It was trained for 10 epochs and achieved a final test accuracy of approximately **97.98%**.

### Results and Visualization

The performance of the model is best visualized through a confusion matrix, which shows the number of correct and incorrect predictions for each digit.

!

### How to Run the Code

To run the code, you'll need a Python environment with TensorFlow/Keras and Matplotlib installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ramlah7/CodeSentinel_NeuralNetwork_MNIST.git
    cd CodeSentinel_NeuralNetwork_MNIST
    ```


2.  **Run the notebook:**
    Open the `neural-network-with-keras.ipynb` file in a Jupyter environment (like Jupyter Notebook or Google Colab) and execute the cells. The script will automatically download and use the MNIST dataset.

---
