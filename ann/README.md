
# Artificial Neural Networks

Basic structures for machine learning and optimization. These `VectorFunction` subclasses are used to approximate intricate
functions using neural networks

## Affine Function

This is the basic linear classifier. It takes a given number of inputs and performs an affine transformation to create its
output. 

    int inputs = 5, outputs = 3;
    VectorFunction linear = new AffineFunction(inputs, outputs);
    
    double[] input = new double[] { ... };
    double[] output = linear.pass(inputs);

To train the function using gradient descent, use the AffineFunctionOptimizer class. These class of optimizers take a batch of
inputs as an array. It also requires the gradient of the loss function with respect to the network outputs. Optimizers can be
created using the static create_optimizer method. The default learning rate is set to 0.001. 

    FunctionOptimizer opt = FunctionOptimizer.create_optimizer(linear);
    opt.set_learning_rate(0.005);

    double[][] inputs = new double[][] { ... };
    double[][] derivatives = new double[][] { ... };

    opt.update_parameters(inputs, derivatives);

## Function Series

To create multilayer networks, use the FunctionSeries class. Any number of VectorFunction objects can be connected in series. 
The create_optimizer method can create a FunctionOptimizer that manages the full chain of functions. 

    VectorFunction series = new FunctionSeries(linear, VectorFunction.softmax);
    FunctionOptimizer opt = FunctionOptimizer.create_optimizer(series);

