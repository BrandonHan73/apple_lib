package apple_lib.function;

/**
 * Represents a function that takes an N-dimensional input and produces an
 * M-dimensional output
 */
public interface VectorFunction {

	/**
	 * Function represented by this object
	 */
	public double[] pass(double... input);

}

