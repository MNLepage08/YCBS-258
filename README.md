# YCBS-258
## Practical Machine Learning

#### 1. Introduction to Deep Learning and Keras:
  - [Deep learning](https://youtu.be/aircAruvnKk) is a sub-field of machine learning that uses algorithms inspired by the structure and function of the brain's (neural networks). 
  - [Perceptron](https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53) is a single layer neural network. Is useful for classifying data sets that are linearly separable (linear binary classifiers). 
  - [Multilayer Perceptron - MLPs: ](https://towardsdatascience.com/multilayer-perceptron-explained-with-a-real-life-example-and-python-code-sentiment-analysis-cb408ee93141)(two or more layers) is called Neural Networks (work the same way of perceptron). Stacking perceptron to from neural network. Optimization through backpropagation. Classifies datasets which are not linearly separable. 
  - [Loss / Cost / Error: ](https://medium.com/artificialis/neural-network-basics-loss-and-cost-functions-9d089e9de5f8)How far off we are from the expected value. How wrong is my neural network (or my model in general).
  - [Activation Function: ](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)Received the sum of product (Inputs * Weights) and map onto a nonlinearity. The activate function give the output of the neuron.
  - [Backpropagation](https://youtu.be/Ilg3gGewQ5U) It is the method of fine-tuning the weights of a neural net based on the error rate obtained in the previous epoch (i.e., iteration). This method helps to calculate the gradient of a loss function with respects to all the weights in the network.
  - [Stochastic Gradient Descent: ](https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31)Before computing error we pass… Batch gradient descent / Mini-batch gradient descent / Stochastic gradient descent.
  - [One-hot Encoding: ](https://youtu.be/v_4KWmkwmsU)Transform our categorical labels into vectors of 0 and 1. Every element will be a zero except for the element that corresponds to the actual category of the given input.
  - [Train, Test & Validation Sets explained](https://youtu.be/Zi-0rlM4RDs)
  - [Overffiting ](https://youtu.be/DEMmkFC6IGM)occurs when our model becomes good at being able to classify or predict on data in the training set but is not as good at classifying data that it wasn't trained on. Unable to generalize well. Addding more data to the training set, Data augmentation, Reduce the complexity of the model, Dropout.
  - [Underfitting:](https://youtu.be/aircAruvnKk) When it's not even able to classify the data it was trained on, let alone data it hasn't seen before. Increase the complexity of the model, Add more features to the input sample, Reduce dropout.
  - [Assignment 1](https://github.com/MNLepage08/YCBS-258/blob/main/Homework_M1_Marie-Noel%20Lepage.ipynb)

#### 2. Hyperparameters and Performance
  - [How To Improve Deep Learning Performance](https://machinelearningmastery.com/improve-deep-learning-performance/)
  - With data: Get more data, [Data augmentation](https://augmentor.readthedocs.io/en/master/), Rescale your data, Transform your data, Feature selection, Reframe your problem.

3. Convolutional Neural Networks
4. Reccurent Neural Networks
5. Representation Learning, Autoencoders and GANs
6. Natural Language Processing
7. Reinforcement Learning
8. Training and Deploying Models at Scale
9. Structuring ML Projects
