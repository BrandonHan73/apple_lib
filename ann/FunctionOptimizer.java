package apple_lib.ann;

import apple_lib.function.VectorFunction;

/**
 * Interface of a basic optimizer. Only considers networks that take a single array as input. Updates using minibatches and
 * requires a gradient as input. 
 */ 
public class FunctionOptimizer {

	////////////////////////////////////////////////////////// FIELDS //////////////////////////////////////////////////////////

	/* Target function */
	protected VectorFunction target;
	 
	/* Learning rate */
	protected double learning_rate;
	protected int training_time;

	/////////////////////////////////////////////////////// CONSTRUCTORS ///////////////////////////////////////////////////////

	/**
	 * Basic constructor. Fixes the input and output sizes.
	 */
	public FunctionOptimizer(VectorFunction func) {
		target = func;

		learning_rate = 0.001;
		training_time = 0;
	}

	////////////////////////////////////////////////////////// STATIC //////////////////////////////////////////////////////////

	public static FunctionOptimizer create_optimizer(VectorFunction func) {
		if(func instanceof AffineFunction) return new AffineFunctionOptimizer((AffineFunction) func);
		if(func instanceof FunctionSeries) return new FunctionSeriesOptimizer((FunctionSeries) func);
		else return new FunctionOptimizer(func);
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
	 * Backpropagates, but does not update any parameters
	 */
	public double[][] update_parameters(double[][] inputs, double[][] deriv) {
		int N = inputs.length;

		double[][] input_deriv = new double[N][];
		for(int item = 0; item < N; item++) {
			double[][] backpropagate = target.backpropagate(inputs[item]);
			input_deriv[item] = new double[inputs[item].length];
			for(int out = 0; out < backpropagate.length; out++) {
				for(int in = 0; in < inputs[item].length; in++) {
					input_deriv[item][in] += deriv[item][out] * backpropagate[out][in];
				}
			}
		}

		return input_deriv;
	}

}

