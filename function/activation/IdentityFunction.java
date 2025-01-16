package apple_lib.function.activation;

import apple_lib.function.DifferentiableScalarFunction;

/**
 * Identity function, where f(x) = x
 */
public class IdentityFunction implements DifferentiableScalarFunction {

	/////////////////////////////// STATIC FIELDS //////////////////////////////

	public final static IdentityFunction implementation = new IdentityFunction();

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public double pass(double input) {
		return input;
	}

	@Override
	public double differentiate(double input, double output) {
		return 1;
	}

}

