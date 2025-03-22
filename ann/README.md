
# Artificial Neural Networks

Basic structures for machine learning and optimization. These `VectorFunction` subclasses are used to approximate intricate
functions using neural networks

## VectorFunction

This is the base class for all neural networks. The `VectorFunction` provides the interface for passing inputs through the
network and determining the gradient at a given input point. 

    VectorFunction func = ... ;
    double[] input = new double[] { ... };

    double[] output = func.pass(input);
    double[][] gradient = func.gradient(input);

The output of the gradient method returns a two dimensional double array. `gradient[i][j]` represents the gradient of the ith
output with respect to the jth input. 

This class also allows multiple inputs to be passed at once. This is used for many different purposes, including multithreading,
batch normalization, and recurrent networks. 

    double[][] inputs = new double[][] { ... };

    double[][] outputs = func.pass_all(inputs);
    double[][][] gradients = func.gradient_all(inputs);

The gradient method only supports functions where the output for a given input is independent of other inputs. Using the batch
gradient method on functions like batch normalization or recurrent networks will throw an exception. 

### ScalarFunction

This is a subclass of the main vector function. It is used for functions where each inputs affects exactly one output. This was
added to speed up gradient and backpropagation calculations. 

    ScalarFunction func = ... ;
    double input = ... ;
    
    double output = func.pass(input);
    double deriv = func.pass(input);

The methods inherited from the vector function class can be used to evaluate multiple inputs at the same time. 

### Standard Functions

Multiple commonly used activation functions have been provided as static fields. 
 - `ScalarFunction.ReLU`
 - `ScalarFunction.softplus`
 - `ScalarFunction.tanh`
 - `ScalarFunction.logistic`
 - `ScalarFunction.swish`
 - `ScalarFunction.loglin`
 - `VectorFunction.softmax`

There are also provided implementations of parameterized functions. Optimization for these functions are described in later
sections. 

    VectorFunction linear = new AffineFunction( input_count, output_count);
    BatchNormalization norm = new BatchNormalization( dimensions );

### Function Composition

For building complex neural networks, these vector functions must be connected in series. The function series class packages
multiple vector functions and passes inputs through each layer sequentially. 

    VectorFunction func = new FunctionSeries( layer1, layer2, ... );

For deeper networks, a residual block implementation is also provided. Users are expected to manage layer sizes. 

    VectorFunction block = new ResidualBlock( func );

## Optimization

Functions with parameters must be learned. The function optimizer class computes derivatives and updates the parameters of
vector functions. To create a vector function, use the provided static method. 

    FunctionOptimizer opt = FunctionOptimizer.create_optimizer( func );

This static method determines what type of function has been provided and instantiates the correct optimizer type. To update
the parameters of the function using this class, provide the inputs and the derivatives of the loss function with respect to
the outputs. 

    double[][] inputs = new double[][] { ... };
    double[][] dloss = new double[][] { ... };

    opt.update_parameters(inputs, dloss);

The parameter update function also performs backpropagation on the provided derivatives and returns the derivative of the loss
function with respect to the network inputs. 

### Classification

Standard loss functions are also implemented. For the classification problem, the classifier optimizer can be used. Instead of
passing the derivatives to the optimizer, pass the correct labels. The optimizer will perform the derivative calculations using
the set loss function. 

    double[][] inputs = new double[][] { ... };
    int[] labels = new int[] { ... };

    ClassifierOptimizer opt = new ClassifierOptimizer( FunctionOptimizer.create_optimizer(func) );
    opt.update_parameters(inputs, labels);

### Hyperparameters

Various learning algorithms have also been implemented. Users can choose which optimization technique to use and set the
respective hyperparameters. 

    opt.use_sgd();
    opt.use_sgd_momentum( momentum );
    opt.use_adagrad( div_protection );
    opt.use_rmsprop( div_protection, decay );
    opt.use_adam( first_bias, second_bias, div_protection );

    opt.set_learning_rate( lr );

Note that these work for both `FunctionOptimizer` and `ClassifierOptimizer`. 

