
# Functions

Basic functions. The `VectorFunction` abstract class provides an interface for a standard layer in a neural network. It takes an
array of values and outputs an array of values. An additional operation is provided for backpropagation. The derivative of each
output with respect to each input is given in a two-dimensional matrix. 

    VectorFunction function = new VectorFunction() { ... };
    double[] input = new double[] { ... };

    double[] activate = function.pass(input);
    double[][] derivative = function.backpropagate(input);
    
The `VectorFunction` class also contains a few static instances of common activation functions. 
 - ReLU
 - Softmax
 - Softplus
 - Hyperbolic tangent
 - Logistic
 - Swish

