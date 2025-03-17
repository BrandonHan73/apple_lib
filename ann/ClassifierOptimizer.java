package apple_lib.ann;

public class ClassifierOptimizer {

	protected FunctionOptimizer optimizer;

	public ClassifierOptimizer(FunctionOptimizer deriv_optimizer) {
		optimizer = deriv_optimizer;
	}

	/**
	 * Sets the optimization strategy to Adam if training has not started yet
	 */
	public void use_adam(double first_moment_bias, double second_moment_bias, double protection) {
		optimizer.use_adam(first_moment_bias, second_moment_bias, protection);
	}

	/**
	 * Sets the optimization strategy to RMSProp if training has not started yet
	 */
	public void use_rmsprop(double protection, double decay) {
		optimizer.use_rmsprop(protection, decay);
	}

	/**
	 * Sets the optimization strategy to ADAGrad if training has not started yet
	 */
	public void use_adagrad(double protection) {
		optimizer.use_adagrad(protection);
	}

	/**
	 * Sets the optimization strategy to SGD+momentum if training has not started yet
	 */
	public void use_sgd_momentum(double decay) {
		optimizer.use_sgd_momentum(decay);
	}

	/**
	 * Sets the optimization strategy to stochastic gradient descent if training has not started yet
	 */
	public void use_sgd() {
		optimizer.use_sgd();
	}

	/**
	 * Sets the learning rate if training has not started yet
	 */
	public void set_learning_rate(double val) {
		optimizer.set_learning_rate(val);
	}

	public double[][] update_parameters(double[][] inputs, int[] labels) {
		int batch_size = inputs.length;
		double[][] outputs = optimizer.target.pass_all(inputs);
		// Calculate derivatives
		double[][] deriv = new double[batch_size][];
		for(int item = 0; item < batch_size; item++) {
			double[] activation = outputs[item];
			deriv[item] = new double[activation.length];
			deriv[item][ labels[item] ] = -1 / (activation[ labels[item] ] + 0.001) / batch_size;
		}

		// Optimize
		return optimizer.update_parameters(inputs, deriv);
	}

}

