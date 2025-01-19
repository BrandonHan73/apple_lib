package apple_lib.ann.base;

/**
 * Defines all functionality expected from an artificial neural network.
 */
public interface ArtificialNeuralNetwork {

	/**
	 * Registers an input without calculating the resulting output
	 */
	public void load_input(double... input);

	/**
	 * Calculates an output without registering the input
	 */
	public double[] test_input(double... input);

	/**
	 * Calculates the output to a specific registered input
	 */
	public double[] calculate(int input);

	/**
	 * Calculates the output to the most recently registered input
	 */
	public double[] calculate();

	/**
	 * Removes all registered inputs
	 */
	public void clear_inputs();

	/**
	 * Specifies a derivative of the cost function with respect to a given input
	 */
	public void load_derivative(int input, double... dCdy);

	/**
	 * Specifies a derivative of the cost function with respect to the most 
	 * recently registered input
	 */
	public void load_derivative(double... dCdy);

	/**
	 * Performs chain rule on the specified output using the loaded derivative
	 */
	public double[] backpropogate(int output);

	/**
	 * Performs chain rule on the most recently registered output using the
	 * loaded derivative
	 */
	public double[] backpropogate();

	/**
	 * Update all parameters using the loaded derivatives for all inputs
	 */
	public void update_parameters();

	/**
	 * Returns the number of inputs currently registered
	 */
	public int data_length();

}

