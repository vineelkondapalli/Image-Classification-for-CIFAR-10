# Image-Classification-for-CIFAR-10

## Project Summary

This project documents the process of building, training, and optimizing a Convolutional Neural Network (CNN) from scratch in PyTorch. The goal was to classify images from the CIFAR-10 dataset, starting with a simple baseline and iteratively improving the model by implementing modern deep learning techniques.

The initial 3-layer CNN quickly began to overfit, which was addressed by integrating a suite of powerful regularization techniques, including **Data Augmentation** (random crops and flips), **Dropout** layers, and **Batch Normalization**.

To push performance further and enable a deeper architecture, the model was redesigned to be ResNet-inspired. This involved implementing **residual (skip) connections** to combat the vanishing gradient problem, a common issue in deeper networks. The final architecture is a custom, wide CNN that uses these residual blocks, followed by a Global Average Pooling layer and a final linear classifier.

The training process itself was optimized with a dynamic **`ReduceLROnPlateau` learning rate scheduler**, which allowed the model to effectively navigate performance plateaus. This iterative process of identifying problems (overfitting, vanishing gradients) and implementing targeted solutions resulted in a final, robust model that achieved **76% accuracy** on the test set.

## Technologies Used

* PyTorch
* TorchVision
* NumPy
* Google Colab & A100 GPU

## Usage

1.  Clone the repository:
    ```bash
    git clone [your-repo-link]
    ```
2.  Install the required dependencies:
    ```bash
    pip install torch torchvision
    ```
3.  Run the main Jupyter Notebook or Python script to train the model and evaluate its performance.
