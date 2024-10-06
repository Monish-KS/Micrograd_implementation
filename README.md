SLP and Backpropagation Implementation from Scratch

This repository contains an implementation of a Single Layer Perceptron (SLP) and the Backpropagation algorithm from scratch using Python and PyTorch. The implementation also includes a unique approach for managing nodes during backpropagation using topological sorting, ensuring an efficient and well-organized computational graph.

Introduction
This project aims to demonstrate how to implement a simple neural network using the fundamentals of machine learning. The key focus is on implementing Single Layer Perceptron (SLP) and Backpropagation from scratch, without relying on PyTorch's automatic differentiation.

Instead of traditional backward graph traversal, this implementation employs topological sorting to manage node computation dependencies during backpropagation. This ensures that all computations are performed in the correct order.

Key Features
1. Implementation of a Single Layer Perceptron (SLP) from scratch.
2. Custom backpropagation mechanism using topological sort for node management.
3. Flexible code structure that is easy to expand to more complex neural networks.
4. Efficient node dependency management via topological sorting.
5. Written in Python with PyTorch support for additional functionalities and optimizations.

Requirements
To run this project, you need to have Python and PyTorch installed. The following libraries are required:

* Python 3.8 or higher
* PyTorch (>= 1.7.0)
* Numpy

'''bash
pip install torch numpy
'''

Explanation
Single Layer Perceptron
A Single Layer Perceptron (SLP) is a type of artificial neural network that consists of a single layer of neurons. The SLP takes input, applies weights, and passes the weighted sum through an activation function (commonly a sigmoid or ReLU) to produce the output.
This project demonstrates how to build an SLP manually using Python and PyTorch to handle the mathematical operations. PyTorch is not used for automatic differentiation here but purely for tensor manipulation.

Backpropagation Algorithm
The Backpropagation algorithm is used to train neural networks by minimizing the loss function. It calculates the gradient of the loss function with respect to each weight by the chain rule and adjusts the weights accordingly.
In this implementation, the backpropagation is computed manually without relying on PyTorch's built-in autograd feature. Instead, the implementation handles the gradient calculations explicitly by traversing the computational graph in reverse order.

Topological Sorting
During backpropagation, it's crucial to update the model's parameters in the correct sequence. In this implementation, topological sorting is employed to manage the order in which the nodes (or neurons) are processed. This ensures that the backpropagation algorithm can efficiently compute gradients by following the dependencies between layers and nodes.

Thanks to @AndrejKarapathy
