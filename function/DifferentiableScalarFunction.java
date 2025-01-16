package apple_lib.function;

/**
 * Represents a differentiable function that takes one input and produces one
 * output
 */
public interface DifferentiableScalarFunction extends ScalarFunction {

	/**
	 * Given an input to this function and the corresponding output, produces
	 * the derivative of this function. Assumes that the this function would
	 * produce the given output when given the input. 
	 */
	public double differentiate(double input, double output);

}

