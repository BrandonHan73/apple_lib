package apple_lib.ann;

import apple_lib.function.ScalarFunction;

/**
 * Interface of a basic optimizer. Only considers networks that take a single array as input. Updates using minibatches and
 * requires a gradient as input. 
 */ 
public class ScalarFunctionOptimizer extends FunctionOptimizer {
	 
	/////////////////////////////////////////////////////// CONSTRUCTORS ///////////////////////////////////////////////////////

	/**
	 * Basic constructor. Fixes the input and output sizes.
	 */
	public ScalarFunctionOptimizer(ScalarFunction func) {
		super(func);
	}

	////////////////////////////////////////////////////////// METHODS /////////////////////////////////////////////////////////

	/**
	 * Backpropagates, but does not update any parameters
	 */
	public double[][] update_parameters(double[][] inputs, double[][] deriv) {
		int N = inputs.length;
		ScalarFunction target = (ScalarFunction) super.target;

		double[][] input_deriv = new double[N][];
		for(int item = 0; item < N; item++) {
			input_deriv[item] = new double[inputs[item].length];
			for(int in = 0; in < inputs[item].length; in++) {
				input_deriv[item][in] += deriv[item][in] * target.backpropagate(inputs[item][in]);
			}
		}

		return input_deriv;
	}

}

