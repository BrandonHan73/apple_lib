package apple_lib.function;

/**
 * Represents a function mapping an n-dimensional input to an m-dimensional output. 
 */
public abstract class VectorFunction {

	/**
	 * Feed forward
	 */
	public abstract double[] pass(double[] input);

	/**
	 * Backpropogate. Derivative of output i with respect to output j is given by output[i][j]. 
	 */
	public abstract double[][] backpropagate(double[] input);

	///////////////////////////////////////////////////// COMMON FUNCTIONS /////////////////////////////////////////////////////

	/* Rectified Linear Unit */
	public final static VectorFunction ReLU = new VectorFunction() {
		@Override
		public double[] pass(double[] input) {
			int N = input.length;
			double[] output = new double[N];
			for(int i = 0; i < N; i++) {
				// Identity if x > 0; Otherwise, return zero
				output[i] = input[i] > 0 ? input[i] : 0;
			}
			return output;
		}
		@Override
		public double[][] backpropagate(double[] input) {
			int N = input.length;
			double[][] output = new double[N][N];
			for(int i = 0; i < N; i++) {
				// Unit if x > 0; Otherwise, return zero
				output[i][i] = input[i] > 0 ? 1 : 0;
			}
			return output;
		}
	};

	/* Softmax */
	public final static VectorFunction softmax = new VectorFunction() {
		@Override
		public double[] pass(double[] input) {
			int N = input.length;
			
			// If input length is zero, then output length is also zero
			if(N == 0) return new double[N];

			// Determine maximum value
			double max = input[0];
			for(int i = 1; i < N; i++) {
				max = input[i] > max ? input[i] : max;
			}

			// Rescale for numerical stability. Exponentiate and take sum. 
			double sum = 0;
			double[] output = new double[N];
			for(int i = 0; i < N; i++) {
				output[i] = Math.exp(input[i] - max);
				sum += output[i];
			}
			
			// Divide entries by sum
			for(int i = 0; i < N; i++) {
				output[i] /= sum;
			}
			return output;
		}
		@Override
		public double[][] backpropagate(double[] input) {
			int N = input.length;
			double[] activation = pass(input);
			double[][] output = new double[N][N];
			for(int out = 0; out < N; out++) {
				for(int in = 0; in < N; in++) {
					// yi ( dirac - yj )
					int dirac = out == in ? 1 : 0;
					output[out][in] = activation[out] * (dirac - activation[in]);
				}
			}
			return output;
		}
	};

	/* Softplus function */
	public final static VectorFunction softplus = new VectorFunction() {
		@Override
		public double[] pass(double[] input) {
			int N = input.length;
			double[] output = new double[N];
			for(int i = 0; i < N; i++) {
				// Create exponential
				output[i] = Math.exp(-input[i]);
				// Check for unbounded value
				if(Double.isInfinite(output[i])) {
					output[i] = input[i];
				} else {
					output[i] = Math.log(1 + output[i]);
				}
			}
			return output;
		}
		@Override
		public double[][] backpropagate(double[] input) {
			int N = input.length;
			double[] sigma = logistic.pass(input);
			double[][] output = new double[N][N];
			for(int i = 0; i < N; i++) {
				output[i][i] = sigma[i];
			}
			return output;
		}
	};

	/* Hyperbolic tangent */
	public final static VectorFunction tanh = new VectorFunction() {
		@Override
		public double[] pass(double[] input) {
			int N = input.length;
			double[] output = new double[N];
			for(int i = 0; i < N; i++) {
				double pos_exp = Math.exp(input[i]);
				double neg_exp = Math.exp(input[i]);
				// Check for unbounded values
				if(Double.isInfinite(pos_exp)) {
					output[i] = 1;
				} else if(Double.isInfinite(neg_exp)) {
					output[i] = -1;
				} else {
					output[i] = (pos_exp - neg_exp) / (pos_exp + neg_exp);
				}
			}
			return output;
		}
		@Override
		public double[][] backpropagate(double[] input) {
			int N = input.length;
			double[] activation = pass(input);
			double[][] output = new double[N][N];
			for(int i = 0; i < N; i++) {
				output[i][i] = 1 - activation[i] * activation[i];
			}
			return output;
		}
	};

	/* Logistic function */
	public final static VectorFunction logistic = new VectorFunction() {
		@Override
		public double[] pass(double[] input) {
			int N = input.length;
			double[] output = new double[N];
			for(int i = 0; i < N; i++) {
				// Create exponential
				output[i] = Math.exp(-input[i]);
				// Check for unbounded value
				if(Double.isInfinite(output[i])) {
					output[i] = 0;
				} else {
					output[i] = 1 / (1 + output[i]);
				}
			}
			return output;
		}
		@Override
		public double[][] backpropagate(double[] input) {
			int N = input.length;
			double[] activation = pass(input);
			double[][] output = new double[N][N];
			for(int i = 0; i < N; i++) {
				output[i][i] = activation[i] * (1 - activation[i]);
			}
			return output;
		}
	};

	/* Swish function */
	public final static VectorFunction swish = new VectorFunction() {
		@Override
		public double[] pass(double[] input) {
			int N = input.length;
			double[] sigma = logistic.pass(input);
			double[] output = new double[N];
			for(int i = 0; i < N; i++) {
				output[i] = sigma[i] * input[i];
			}
			return output;
		}
		@Override
		public double[][] backpropagate(double[] input) {
			int N = input.length;
			double[] sigma = logistic.pass(input);
			double[][] sigma_deriv = logistic.backpropagate(input);
			double[][] output = new double[N][N];
			for(int i = 0; i < N; i++) {
				output[i][i] = sigma[i] + input[i] * sigma_deriv[i][i];
			}
			return output;
		}
	};

	/* Half-logarithmic half-linear function */
	public final static VectorFunction loglin = new VectorFunction() {
		@Override
		public double[] pass(double[] input) {
			int N = input.length;
			double[] output = new double[N];
			for(int i = 0; i < N; i++) {
				output[i] = input[i] >= 0 ? input[i] : -Math.log(1 - input[i]);
			}
			return output;
		}
		@Override
		public double[][] backpropagate(double[] input) {
			int N = input.length;
			double[][] output = new double[N][N];
			for(int i = 0; i < N; i++) {
				output[i][i] = input[i] >= 0 ? 1 : 1 / (1 - input[i]);
			}
			return output;
		}
	};

}

