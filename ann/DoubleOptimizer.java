package apple_lib.ann;

/**
 * Simulates the trajectory of a single double value. Used for optimization. 
 */
public class DoubleOptimizer {

	////////////////////////////////////////////////////////// FIELDS //////////////////////////////////////////////////////////

	/* Learning rate */
	protected double learning_rate;
	protected int training_time;

	/* Training strategy */
	public static enum Strategy {
		SGD, SGD_momentum, ADAGrad, RMSProp, Adam
	}
	protected Strategy optimizer;

	/* SGD+momentum parameters */
	protected double sgdm_decay;

	/* ADAGrad parameters */
	protected double adagrad_protection;

	/* RMSProp parameters */
	protected double rms_protection, rms_decay;

	/* Adam parameters */
	protected double adam_first_bias, adam_second_bias, adam_protection;

	/* Update parameters */
	protected double first_moment, second_moment;

	/////////////////////////////////////////////////////// CONSTRUCTORS ///////////////////////////////////////////////////////

	/**
	 * Basic constructor. Fixes the input and output sizes.
	 */
	public DoubleOptimizer() {
		optimizer = Strategy.Adam;
		
		learning_rate = 0.001;
		training_time = 0;

		sgdm_decay = 0.99;

		adagrad_protection = 0.01;

		rms_protection = 0.01;
		rms_decay = 0.9;

		adam_first_bias = 0.9;
		adam_second_bias = 0.99;
		adam_protection = 0.01;

		first_moment = 0;
		second_moment = 0;
	}

	////////////////////////////////////////////////////////// METHODS /////////////////////////////////////////////////////////

	/**
	 * Sets the optimization strategy to Adam if training has not started yet
	 */
	public void use_adam(double first_moment_bias, double second_moment_bias, double protection) {
		if(training_time == 0) {
			optimizer = Strategy.Adam;
			adam_first_bias = first_moment_bias;
			adam_second_bias = second_moment_bias;
			adam_protection = protection;
		}
	}

	/**
	 * Sets the optimization strategy to RMSProp if training has not started yet
	 */
	public void use_rmsprop(double protection, double decay) {
		if(training_time == 0) {
			optimizer = Strategy.RMSProp;
			rms_protection = protection;
			rms_decay = decay;
		}
	}

	/**
	 * Sets the optimization strategy to ADAGrad if training has not started yet
	 */
	public void use_adagrad(double protection) {
		if(training_time == 0) {
			optimizer = Strategy.ADAGrad;
			adagrad_protection = protection;
		}
	}

	/**
	 * Sets the optimization strategy to SGD+momentum if training has not started yet
	 */
	public void use_sgd_momentum(double decay) {
		if(training_time == 0) {
			optimizer = Strategy.SGD_momentum;
			sgdm_decay = decay;
		}
	}

	/**
	 * Sets the optimization strategy to stochastic gradient descent if training has not started yet
	 */
	public void use_sgd() {
		if(training_time == 0) {
			optimizer = Strategy.SGD;
		}
	}

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
	 * Takes the derivatives of the loss function with respect to the target. Calculates update using set learning strategy.
	 */
	public double calculate_update(double deriv) {
		training_time++;
		double learning_rate = get_learning_rate();
		switch (optimizer) {
			case SGD:
			return -learning_rate * deriv;

			case SGD_momentum:
			first_moment = sgdm_decay * first_moment + deriv;
			return -learning_rate * first_moment;

			case ADAGrad:
			second_moment += deriv * deriv;
			return -learning_rate * deriv / (Math.sqrt(second_moment) + adagrad_protection);

			case RMSProp:
			second_moment = rms_decay * second_moment + (1 - rms_decay) * deriv * deriv;
			return -learning_rate * deriv / Math.sqrt(second_moment) + rms_protection;

			case Adam:
			first_moment = adam_first_bias * first_moment + (1 - adam_first_bias) * deriv;
			second_moment = adam_second_bias * second_moment + (1 - adam_second_bias) * deriv * deriv;
			double first_adjust = first_moment / (1 - Math.pow(adam_first_bias, training_time));
			double second_adjust = second_moment / (1 - Math.pow(adam_second_bias, training_time));
			return -learning_rate * first_adjust / (Math.sqrt(second_adjust) + adam_protection);

			default:
			return 0;
		}
	}

}

