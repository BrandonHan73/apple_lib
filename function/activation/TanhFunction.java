package apple_lib.function.activation;

import apple_lib.function.DifferentiableScalarFunction;

/**
 * Hyperbolic tangent function, where 
 *         exp(x) - exp(-x)
 * f(x) = ------------------
 *         exp(x) + exp(-x)
 */
public class TanhFunction implements DifferentiableScalarFunction {

	/////////////////////////////// STATIC FIELDS //////////////////////////////

	public final static TanhFunction implementation = new TanhFunction();

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public double pass(double input) {
		return Math.tanh(input);
	}

	@Override
	public double differentiate(double input, double output) {
		return 1 - output * output;
	}

}

