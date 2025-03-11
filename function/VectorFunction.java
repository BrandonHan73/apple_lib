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

	/* Scalar functions */
	public final static VectorFunction ReLU = ScalarFunction.ReLU;
	/* Softplus function */
	public final static VectorFunction softplus = ScalarFunction.softplus;
	/* Hyperbolic tangent */
	public final static VectorFunction tanh = ScalarFunction.tanh;
	/* Logistic function */
	public final static VectorFunction logistic = ScalarFunction.logistic;
	/* Swish function */
	public final static VectorFunction swish = ScalarFunction.swish;
	/* Half-logarithmic half-linear function */
	public final static VectorFunction loglin = ScalarFunction.loglin;

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

}

