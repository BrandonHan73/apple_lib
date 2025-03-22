package apple_lib.function;

/**
 * Subclass of a vector function. Allows elementwise operations to be defined more easily. 
 */
public abstract class ScalarFunction extends VectorFunction {

	@Override
	public double[] pass(double[] input) {
		double[] out = new double[input.length];
		for(int i = 0; i < out.length; i++) {
			out[i] = pass(input[i]);
		}
		return out;
	}

	@Override
	public double[][] gradient(double[] input) {
		double[][] out = new double[input.length][input.length];
		for(int i = 0; i < out.length; i++) {
			out[i][i] = gradient(input[i]);
		}
		return out;
	}

	/**
	 * Feed forward
	 */
	public abstract double pass(double input);

	/**
	 * Calculates the derivative
	 */
	public abstract double gradient(double input);

	///////////////////////////////////////////////////// COMMON FUNCTIONS /////////////////////////////////////////////////////

	/* Rectified Linear Unit */
	public final static ScalarFunction ReLU = new ScalarFunction() {
		@Override
		public double pass(double input) {
			// Identity if x > 0; Otherwise, return zero
			return input > 0 ? input : 0;
		}
		@Override
		public double gradient(double input) {
			return input > 0 ? 1 : 0;
		}
	};

	/* Softplus function */
	public final static ScalarFunction softplus = new ScalarFunction() {
		@Override
		public double pass(double input) {
			// Create exponential
			double exp = Math.exp(-input);
			// Check for unbounded value
			return Double.isInfinite(exp) ? input : Math.log(1 + exp);
		}
		@Override
		public double gradient(double input) {
			return logistic.pass(input);
		}
	};

	/* Hyperbolic tangent */
	public final static ScalarFunction tanh = new ScalarFunction() {
		@Override
		public double pass(double input) {
			double pos_exp = Math.exp(input);
			double neg_exp = Math.exp(input);
			// Check for unbounded values
			return Double.isInfinite(pos_exp) ? 1 : Double.isInfinite(neg_exp) ? -1 : (pos_exp - neg_exp) / (pos_exp + neg_exp);
		}
		@Override
		public double gradient(double input) {
			double activation = pass(input);
			return 1 - activation * activation;
		}
	};

	/* Logistic function */
	public final static ScalarFunction logistic = new ScalarFunction() {
		@Override
		public double pass(double input) {
			// Create exponential
			double exp = Math.exp(-input);
			// Check for unbounded value
			return Double.isInfinite(exp) ? 0 : 1 / (1 + exp);
		}
		@Override
		public double gradient(double input) {
			double activation = pass(input);
			return activation * (1 - activation);
		}
	};

	/* Swish function */
	public final static ScalarFunction swish = new ScalarFunction() {
		@Override
		public double pass(double input) {
			return logistic.pass(input) * input;
		}
		@Override
		public double gradient(double input) {
			return logistic.pass(input) + input * logistic.gradient(input);
		}
	};

	/* Half-logarithmic half-linear function */
	public final static ScalarFunction loglin = new ScalarFunction() {
		@Override
		public double pass(double input) {
			return input >= 0 ? input : -Math.log(1 - input);
		}
		@Override
		public double gradient(double input) {
			return input >= 0 ? 1 : 1 / (1 - input);
		}
	};

}

