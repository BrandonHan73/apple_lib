package apple_lib.ann;

import apple_lib.function.ScalarFunction;
import apple_lib.function.VectorFunction;

/**
 * Interface of a basic optimizer. Only considers networks that take a single array as input. Updates using minibatches and
 * requires a gradient as input. 
 */ 
public class FunctionOptimizer {

	////////////////////////////////////////////////////////// FIELDS //////////////////////////////////////////////////////////

	/* Target function */
	protected VectorFunction target;

	/* Unit optimizers */
	protected DoubleOptimizer[] optimizers;
	 
	/////////////////////////////////////////////////////// CONSTRUCTORS ///////////////////////////////////////////////////////

	/**
	 * Basic constructor. Fixes the input and output sizes.
	 */
	public FunctionOptimizer(VectorFunction func) {
		target = func;

		optimizers = new DoubleOptimizer[0];
	}

	////////////////////////////////////////////////////////// STATIC //////////////////////////////////////////////////////////

	public static FunctionOptimizer create_optimizer(VectorFunction func) {
		if(func instanceof AffineFunction) return new AffineFunctionOptimizer((AffineFunction) func);
		if(func instanceof FunctionSeries) return new FunctionSeriesOptimizer((FunctionSeries) func);
		if(func instanceof ResidualBlock) return new ResidualBlockOptimizer((ResidualBlock) func);
		if(func instanceof ScalarFunction) return new ScalarFunctionOptimizer((ScalarFunction) func);
		else return new FunctionOptimizer(func);
	}

	////////////////////////////////////////////////////////// METHODS /////////////////////////////////////////////////////////

	/**
	 * Sets the optimization strategy to Adam if training has not started yet
	 */
	public void use_adam(double first_moment_bias, double second_moment_bias, double protection) {
		for(DoubleOptimizer opt : optimizers) {
			opt.use_adam(first_moment_bias, second_moment_bias, protection);
		}
	}

	/**
	 * Sets the optimization strategy to RMSProp if training has not started yet
	 */
	public void use_rmsprop(double protection, double decay) {
		for(DoubleOptimizer opt : optimizers) {
			opt.use_rmsprop(protection, decay);
		}
	}

	/**
	 * Sets the optimization strategy to ADAGrad if training has not started yet
	 */
	public void use_adagrad(double protection) {
		for(DoubleOptimizer opt : optimizers) {
			opt.use_adagrad(protection);
		}
	}

	/**
	 * Sets the optimization strategy to SGD+momentum if training has not started yet
	 */
	public void use_sgd_momentum(double decay) {
		for(DoubleOptimizer opt : optimizers) {
			opt.use_sgd_momentum(decay);
		}
	}

	/**
	 * Sets the optimization strategy to stochastic gradient descent if training has not started yet
	 */
	public void use_sgd() {
		for(DoubleOptimizer opt : optimizers) {
			opt.use_sgd();
		}
	}

	/**
	 * Sets the learning rate if training has not started yet
	 */
	public void set_learning_rate(double val) {
		for(DoubleOptimizer opt : optimizers) {
			opt.set_learning_rate(val);
		}
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

