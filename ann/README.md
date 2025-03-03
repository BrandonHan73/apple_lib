
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

To train the function using gradient descent, use the `AffineFunctionOptimizer` class. These class of optimizers take a batch of
inputs as an array. It also requires the gradient of the loss function with respect to the network outputs. Optimizers can be
created using the static `create_optimizer` method. The default learning rate is set to 0.001. 

    FunctionOptimizer opt = FunctionOptimizer.create_optimizer(linear);
    opt.set_learning_rate(0.005);

    double[][] inputs = new double[][] { ... };
    double[][] derivatives = new double[][] { ... };

    opt.update_parameters(inputs, derivatives);

There is also the option to set the optimization algorithm. The following algorithms are supported. 
 - Stochastic gradient descent: `opt.use_sgd();`
 - Stochastic gradient descent with momentum: `opt.use_sgd_momentum(decay);`
 - ADAGrad: `opt.use_adagrad(min_denominator);`
 - RMSProp: `opt.use_rmsprop(min_denominator, decay);`
 - Adam: `opt.use_adam(first_moment_decay, second_moment_decay, min_denominator);`
Here, `min_denominator` is added to any division operation to ensure no division by zero occurs. The default algorithm is Adam
with parameters 0.9, 0.99, and 0.00000001. 

## Function Series

To create multilayer networks, use the `FunctionSeries` class. Any number of `VectorFunction` objects can be connected in 
series. The create_optimizer method can create a `FunctionOptimizer` that manages the full chain of functions. 

    VectorFunction series = new FunctionSeries(linear, VectorFunction.softmax);
    FunctionOptimizer opt = FunctionOptimizer.create_optimizer(series);

For deeper networks, you may want to use residual blocks. A residual block can be made using any vector function. 

    VectorFunction block = new ResidualBlock(series);

## Loss Functions

Loss functions automatically calculate gradients and passes them into optimizers. Classifier optimizers are used for functions
where outputs are bounded between 0 and 1. 

    VectorFunction classifier = ...;
    FunctionOptimizer optimizer = FunctionOptimizer.create_optimizer(classifier);
    ClassifierOptimizer loss = new ClassifierOptimizer(optimizer);

    double[][] items = ...;
    int[] labels = ...;
    loss.update_parameters(items, labels);

By default, the `ClassifierOptimizer` class uses the cross entropy loss. 

