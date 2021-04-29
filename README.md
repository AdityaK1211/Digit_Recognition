# Digit_Recognition
### Digit recognition using Tensorflow 2.0, Keras and Pygame for prediction of input digit.
Dataset used: MNIST

#### Run app.py for writing a digit as input in the Pygame window to get the predicted digit from the trained model.

Classify hand-written digit into appropriate class using machine learning algorithm.  Use MNIST dataset to create the model. 

MNIST Dataset: http://yann.lecun.com/exdb/mnist/

## Introduction 
The handwritten digit recognition is the ability of computers to recognize human handwritten 
digits. It is a hard task for the machine because handwritten digits are not perfect and can be 
made with many different flavours. The handwritten digit recognition is the solution to this 
problem which uses the image of a digit and recognizes the digit present in the image. It is used 
in different tasks of our real-life time purposes. Precisely, it is used in vehicle number plate 
detection, banks for reading checks, post offices for sorting letter, and many other related 
tasks. We are going to implement a handwritten digit recognition app using the MNIST dataset. 
We will be using a special type of deep neural network that is Convolutional Neural Networks
(CNN). We have built a GUI in which you can draw the digit and recognize it.

## Dataset 
This is probably one of the most popular datasets among machine learning and deep learning 
enthusiasts. The MNIST dataset contains 60,000 training images of handwritten digits from 
zero to nine and 10,000 images for testing. So, the MNIST dataset has 10 different classes. The 
handwritten digits images are represented as a 28×28 matrix where each cell contains 
grayscale pixel value

## Structure of Neural Network 
A neural network is made up by stacking layers of neurons, and is defined by the weights of 
connections and biases of neurons. Activations are a result dependent on a certain input.
This structure is known as a feedforward architecture because the connections in the network 
flow forward from the input layer to the output layer without any feedback loops. In this figure:<br>
▪ The input layer contains the predictors.<br>
▪ The hidden layer contains unobservable nodes, or units. The value of each hidden unit is 
some function of the predictors; the exact form of the function depends in part upon the 
network type and in part upon user-controllable specifications.<br>
▪ The output layer contains the responses. Since the history of default is a categorical 
variable with two categories, it is recoded as two indicator variables. Each output unit is 
some function of the hidden units. Again, the exact form of the function depends in part 
on the network type and in part on user-controllable specifications

## Conclusion 
The performance of the classifier can be measured in terms of ability to identify a condition 
properly (sensitivity), the proportion of true results (accuracy), number of positive results from 
the procedure of classification as false positives (positive predictions) and ability to exclude 
condition correctly (specificity). The training accuracy and testing accuracy achieved is 99.92%
and 99.16% respectively using Convolutional Neural Network (CNN).

## References 
1. http://yann.lecun.com/exdb/mnist/
2. Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner “Gradient‐Based Learning 
Applied to Document Recognition”, IEEE, November 1998
3. Very Deep Convolutional Networks for Large-Scale Image Recognition 
https://arxiv.org/abs/1409.1556
4. Narender Kumar, Himanshu Beniwal, “Survey on Handwritten Digit Recognition using 
Machine Learning”, IJCSE, June 2018
