# Artificial Neural Networks 2.0

Centralizes structure around the input history to hopefully allow for more
parallelism and advanced structures. 

## ArtificialNeuralNetwork interface

Use builders or constructors to create an ArtificialNeuralNetwork. All network
types implement this interface. 

    ArtificialNeuralNetwork network = ...

### Passing inputs

There exist two ways to apply an input to the network. 

The first way is to apply an input without adding it to the input history. The
network will calculate and return the output immediately. 

    double[] output = network.test_input(input);

For networks that do not depend on past inputs, this would be the most effective
way for passing data because the network does not need to keep records of the
activation. 

The second way to apply an input is to add it to the history and poll the output
separately. 

    network.load_input(input);
    double[] output = network.calculate();

This method should be used for networks that depend on past inputs. Using a
separate function call to obtain the network output allows all calculations to
occur at the same time. 

    for(double[] input_i : input_data) {
        network.load_input(input_i);
    }
    double[] output = network.calculate();

Users may also determine the output for past inputs by specifying the input
index. If no index is specified, then the network will return the output of the
most recent input. 
    
    double[] output_i = network.calculate(i);

Input history can be cleared. However the full history must be cleared at once.
Users cannot remove individual input records. 

    network.clear_inputs();

Users can determine the total number of inputs currently loaded into the
network.

    int input_count = network.data_length();

### Backpropogation

To use backpropogation, users must specify the derivative of the cost function
with respect to the network output by supplying the derivative and the output
index. 

    network.load_derivative(index, dCdy);

If no index is given, then the network will assume the derivative applies to the
most recent output. Output indices correspond to the respective input indices. 
If multiple derivatives are loaded to the same output, the derivatives will be
combined using addition. 

Backpropogate to perform a chain rule calculation of the loaded derivatives. 

    double[] dCdx = network.backpropogate(index);

The calculation will use the loaded derivative values and the recorded
activation values for the specifies index. The returned value will be the
derivative of the cost function with respect to the specified network input. 
**Weights and biases are not updated by this method.** If an index is not
specified, calculations will be performed on the most recent input.

To update weights and biases, use the corresponding method. 

    ANN_Base.gradient_descent(network);

Calling this method will update all parameters based off of the current records
and loaded derivatives. After updates are complete, certain data will become
outdated and will be cleared. 
 - Output records. Changing the parameters would affect the calculated outputs. 
 - Loaded derivatives. The changed output will affect the cost function, so the
   derivative of the cost function with respect to the new parameters are
   unlikely to remain the same. 
These will have to be recalculated. Importantly, the input values will not be
removed from the record. 

A static method is used to apply gradient descent because updates cannot be
efficiently performed from an instance-based perspective. The non-static method
update_parameters in the ArtificialNeuralNetwork interface performs gradient
descent on a single frame without unloading data. 

