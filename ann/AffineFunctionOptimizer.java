package apple_lib.ann;

/**
 * Optimizer for an affine function. 
 */ 
public class AffineFunctionOptimizer {

	////////////////////////////////////////////////////////// FIELDS //////////////////////////////////////////////////////////
	 
	/* Function to optimize */
	protected AffineFunction function;

	/* Learning rate */
	protected double learning_rate;
	protected int training_time;

	/////////////////////////////////////////////////////// CONSTRUCTORS ///////////////////////////////////////////////////////

	/**
	 * Basic constructor. Fixes the input and output sizes.
	 */
	public AffineFunctionOptimizer(AffineFunction target) {
		function = target;

		learning_rate = 0.001;
		training_time = 0;
	}

	////////////////////////////////////////////////////////// METHODS /////////////////////////////////////////////////////////

	/**
	 * Sets the learning rate if training has not started yet
	 */
	public void set_learning_rate(double val) {
		if(training_time == 0) {
			learning_rate = val;
		}
	}

	/**
	 * Determines the current learning rate based on the calculation algorithm and the set hyperparameters. 
	 */
	protected double get_learning_rate() {
		return learning_rate;
	}

	/**
	 * Updates the parameters of the affine function using the given inputs and derivatives. 
	 */
	public double[][] update_parameters(double[][] inputs, double[][] deriv) {
		int N = inputs.length;

		// Determine derivatives with respect to inputs and parameters
		double[][] param_deriv = new double[function.input_count + 1][function.output_count];
		double[][] input_deriv = new double[N][];
		for(int item = 0; item < N; item++) {
			double[][] backpropagate = function.backpropagate(inputs[item]);
			input_deriv[item] = new double[function.input_count];
			for(int out = 0; out < function.output_count; out++) {
				for(int in = 0; in < function.input_count; in++) {
					param_deriv[in][out] += deriv[item][out] * inputs[item][in];
					input_deriv[item][in] += deriv[item][out] * backpropagate[out][in];
				}
				param_deriv[function.input_count][out] += deriv[item][out];
			}
		}

		// Update training time and determine learning rate
		training_time++;
		double learning_rate = get_learning_rate();

		// Update parameters
		for(int in = 0; in <= function.input_count; in++) {
			for(int out = 0; out < function.output_count; out++) {
				function.parameters[in][out] -= learning_rate * param_deriv[in][out];
			}
		}

		return input_deriv;
	}

}

