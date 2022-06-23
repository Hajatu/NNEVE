# Neural network training
## Introduction
There are 3 methods to train a neural network (train_generations, train and
train_step). 
### `#!python def train_generations(self, params: QOParams, generations, epochs, plot) -> Generator["QONetwork", None, None]`
In train_generations the generation of test trials is defined.
### `#!python def train(self, params: QOParams, epochs)`
The train function is used to train the neural network on a given number of epochs.
At the end of each test trial, a copy of the model containing the parameters
(e.g., weights) of the trained model is returned.

### `#!python def train_step(self, x: tf.Tensor, params: QOParams) -> tf.Tensor`
Train_step implements the individual step of learning the network using the previously
implemented loss function. Train step returns the average loss value of the
neural network for the given problem. You can manipulate the behavior of the
network when training it.
## Optimization
Parameters that we can optimize are: 
###generations -> number of generations of training trials,
###epochs -> number of epochs of training the network,
###plot -> auxiliary parameter for retaining individual neural network learning values