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
  - [Normalization & Standardization](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)
  - [Assignment 1](https://github.com/MNLepage08/YCBS-258/blob/main/Homework_M1_Marie-Noel%20Lepage.ipynb)

#### 2. Hyperparameters and Performance
  - [How To Improve Deep Learning Performance](https://machinelearningmastery.com/improve-deep-learning-performance/)
  - With data: Get more data, [Data augmentation](https://augmentor.readthedocs.io/en/master/), Rescale your data, Transform your data, Feature selection, Reframe your problem.
  - With Algorithm: Maybe you chosen algoritm is not the best for your problem, Published research is highly optimized, Resampling mothods (k-fold cross validation, make the dataset smaller and use strong resampling methods).
  - With Algoritm Tuning: Diagnostics (overfitting/underfitting), Weight initialisation, Learning rate & scedulling ([cyclical learning rates with keras and DL](https://pyimagesearch.com/2019/07/29/cyclical-learning-rates-with-keras-and-deep-learning/#:~:text=learning%20rate%20range.-,What%20are%20cyclical%20learning%20rates%3F,you%20simply%20need%20a%20callback)), Activation function, Network topology, Batches & Epochs, Regularization (dropout, batch nomalization, Ridge & Lasso), Optimization & Loss, Early Stopping.
  - With Ensembles: Combine models, Combine views, Stacking.
  - How to choose Hyperparameters: Selecting by hand, [Grid Search](https://medium.com/fintechexplained/what-is-grid-search-c01fe886ef0a), [Random Search](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf), [Bayesian Optimization](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f)
  - [A Recipe for Training Networks](http://karpathy.github.io/2019/04/25/recipe/): Become one with data, Set up the end to end training / evaluation skeleton + get dump baseline, Overfit, Regularize, Tune, Squeeze out the juice.
  - [Assignment 2](https://github.com/MNLepage08/YCBS-258/blob/main/Homework_M2_Marie-Noel%20Lepage.ipynb)
  
#### 3. Convolutional Neural Networks
  - [CNN ](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) is a neural network that has one or more convolutional layers and are used mainly for image processing, classification, segmentation and also for other auto correlated data. The role of the CNN is to reduce the images into a form which is easier to process, without losing features which are critical for getting a good prediction.
  - Convolution Layer – The kernel: Apply filters to extract features, Filters are composed of small kernels learned, One bias per filter, Apply activation function on every value of feature map.
  - The objective of the Convolution Operation is to extract the high-level features such as edges, from the input image.  There are two types of results to the operation: Valid Padding: the convolved feature is reduced in dimensionality as compared to the input. Same Padding: the dimensionality is either increased or remains the same. 
  - Pooling Layer: You don’t want small detail (reduce the risk of overfitting). Reduce dimmensianlity, Extract maximum of average region, Sliding window approach. Max Pooling returns the maximum value from the portion of the image covered by the Kernel. Average Pooling returns the average of all the values from the portion of the image covered by the Kernel. 
  - Strides: is the step size that take at every convolutional computation. (horizontal, vertical).
  - Classification – Fully Connected Layer (FC Layer): You try to combine your tensor in the way of your conventional ML need. Aggregate information from final feature maps. Generate final classification
  - [Assignment 3](https://github.com/MNLepage08/YCBS-258/blob/main/Homework_M3_Marie_Noel_Lepage.ipynb)

#### 4. Reccurent Neural Networks
  - [RNN: ](https://towardsdatascience.com/recurrent-neural-networks-d4642c9bc7ce)Sequence of data points that occur in successive order over some period of time. You want to predict the future. Sequence of the event connected to each other’s. Each value related to the lag version.
  - 2 classes: Systematic: components have consistency or occurrence and can be described and modeled (this course). Non-Systematic: Components cannot be directly modeled.
  - [Components for sequence modeling: ](https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/)Level, Trend, Seasonality, Noise (residual).
  - Combining time series components: Additive model & Multiplicative model.
  - Autocorrelation: Represents the degree of similarity between a given time series and a lagged version of itself over successive time intervals. Measures the relationship between a variable’s current value and its past values. Identify seasonality and trend in time series data.
  - [5 differents DL architecture for time series: ](https://towardsdatascience.com/time-series-forecasting-with-deep-learning-and-attention-mechanism-2d001fc871fc) RNNs, LSTM, GRU, Encoder-Decoder model, Attention mechanism.
  - [Assignment 4](https://github.com/MNLepage08/YCBS-258/blob/main/Homework_M4_Marie_Noel_Lepage_v2.ipynb)

#### 5. Representation Learning, Autoencoders and GANs
  - [Autoencoders: ](https://towardsdatascience.com/deep-inside-autoencoders-7e41f319999f)Work by compressing the input into latent-space representation (keep the best features) and then reconstructing the output from this representation.
  - Used for Dimensionality reduction, image compression, image denoising, feature extraction, image generation (GANs), seq to seq prediction (LSTM), recommendation system, anomaly detection (outliers).
  - Encoder: Compresses the input into a latent-space representation.
  - Decoder: Aims to reconstruct the input from the latent space representation.
  - Undercomplete: to constrain h to have smaller dimension that x. We force the autoencoder to learn the most salient features of the training data.
  - Overcomplete: The dimension of the latent representation is greater than the input. Learn to copy the input to the output without learning anything useful about the data distribution.
  - Types of autoencoder: Vanilla, Multilayer, Convolutional, [Regularized](https://keras.io/api/layers/regularizers/).
  - [Variational autoencoders - VAE: ](https://www.jeremyjordan.me/variational-autoencoders/)provides a probabilistic manner for describing an observation in latent space. We’ll formulate our encoder to describe a probability distribution for each latent attribute. The main benefit is that we're capable of learning smooth latent state representations of the input data.
  - [GANs: ](https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29)Generator starts of generating noise. Discriminator tries to predict if an input is real or fake.
  - [LSTM (seq2seq): ](https://machinelearningmastery.com/lstm-autoencoders/): The length of the input sequence can vary.	The temporal ordering of the observations can make it challenging to extract features suitable for use as input to supervised learning models, often requiring deep expertise in the domain or in the field of signal processing.
  - [Tied Weights: ](https://medium.com/@lmayrandprovencher/building-an-autoencoder-with-tied-weights-in-keras-c4a559c529a2)this is a form of parameter sharing, which reduces the number of parameters of the model. Advantages include increased training speed and reduced risk of overfitting. Common practice when building a symmetrical autoencoder.


6. Natural Language Processing
7. Reinforcement Learning
8. Training and Deploying Models at Scale
9. Structuring ML Projects
