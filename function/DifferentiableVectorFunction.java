package apple_lib.function;

/**
 * Represents a differentiable function that takes an N-dimensional input and
 * produces an M-dimensional output
 */
public interface DifferentiableVectorFunction extends VectorFunction {

	/**
	 * Given an input to this function and the corresponding output, produces
	 * the derivative of this function. Assumes that the this function would
	 * produce the given output when given the input. 
	 *
	 * Returns a matrix out[][]. Each out[i][j] corresponds to the partial
	 * derivative dy_i / dx_j, where y_i is the ith output and x_j is the
	 * jth input. 
	 */
	public double[][] differentiate(double[] input, double[] output);

}

