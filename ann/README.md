
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



