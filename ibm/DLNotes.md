# Deep Learning

Neural networks can find complex patterns. Deep nets break complex patterns down into smaller ones to use as building blocks.</br>

### Choosing a Deep Network
Unsupervised Learning: Restricted Boltzmann Machine, Autoencoder</br>

#### Labeled Data
1. Text Processing: Recurrent net, Recursive Neural Tensor Net</br>
2. Image Recognition: Deep Belief Network, Convolutional Network</br>
3. Object Recognition: Convolutional Net, Recurrent Neural Tensor Net</br>
4. Speech Recognition: Recurrent Net</br>
5. Classification:  Multi-layer Perceptrons (MLP/RELU), Deep Belief Network</br>
6. Time Series Analysis: Recurrent Net

### Neural Networks
Interconnected network of nodes.  Neural networks are highly structured. </br>

Forward Propagation: Each input node has its own classifier that is activated and passed forward until the output layer is reached.  Each set of inputs is modified at each label with weights and biases as it moves through layers.</br>

Training the network is done with using inputs with known outputs- the weights and biases are modified to make the network produce accurate results.  It can then be used on new inputs with unknown outputs.</br>

### Classifier
Classifiers produce a score to show the level of confidence in the classification of an item.</br>

### Backpropagation
Vanishing/Exploding Gradient Problem.</br>
During training the cost is being constantly calculated (cost = predicted-actual output) the weights and biases are adjusted to reduce the cost. </br>
The gradient measures the rate of change due to a change in weight or bias. When the gradient is large, the net will train quickly, when it is small, it will train slowly.  The early layers are slow to train- and errors in the early layers can propagate through the whole network.</br>
Backpropagation calculates the gradients from the output to input layers, multiplying all the gradients from the previous layers. This can lead to exponentially small gradients (multiplying numbers between 0 and 1).</br>  

### Restricted Boltzmann Machines (RBM)
Overcome vanishing gradient problem with RBM.  An RBM is a shallow net with two layers: visible and hidden.  No nodes are connected within a layer. In the forward pass the inputs are translated to numbers and in the backwards pass, they are translated back.  If the network is well trained, it will be accurate.  The weights and biases allow the RBM to determine the relationships between the input features and find the important inputs for detecting patterns.  </br>
Feature extraction, autoencoding.

### Deep Belief Nets (DBN)
Alternative to backpropagation.  The difference between a DBN and a multilayer perceptron is the training method.  Is a stack of RBMs. The training time is short, it is accurate and it only needs a small labeled training set.

### Convolutional Nets (CN)
(Machine vision applications)</br>
Convolutional layer: uses convolution to search for a pattern.  Weights and biases affect this step. Neurons perform the convolution instead of an activation.  The neurons are only connected to a set of inputs.  </br>
RELU and Pooling: Rectified Linear Unit, vanishing gradient can be a problem- so relu keeps the gradient at the same level or all layers.  The pooling layer reduces the dimensionality for the most relevant patterns.</br>
Once the pooling is done, then a fully connected layer does the classification.</br>
The do need a large set of labeled data for training.

### Recurrent Nets (RN)
Used for time series and patterns that change over time.  It has a built in feedback loop that makes it good for forecasting.  Can use a sequence of values as inputs and can output a sequence.  Can be stacked to produce better results, and use backpropagation that can be exponentially worse.  This can be solved by gating, gradient clipping, better optimizers.

### Autoencoders
A type of neural network that uses unlabled inputs, encodes them, the reconstructs them.  It acts as a feature extraction engine- finding the most important features.  Usually have 3 layers: input, hidden, and output.  Uses backpropagation with loss (how much info was lost during the reconstruction).  Can be deep or shallow and work well for dimensionality reduction.  Random noise can be used to improve funtionality. Deep autoencoder works better than priciple component analysis.

### Recursive Neural Tensor Nets
Discover the hierarchical nature of the information.  Good for sentiment analysis, uses the order of the words.  RNTN has roots and leaves, built like a tree. Uses labeled training data- compares result to expected sentence structure.  Used for sentiment analysis, parsing images into components.

### Use Cases
Machine vision: image classification, object recognition, video recognition
Fact Extraction
Translation
Sentiment Analysis



